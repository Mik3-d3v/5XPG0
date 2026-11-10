// ann.c — ANN (Linear-ReLU-...-Linear) inference on PicoRV32, the
// counterpart of the SNN in main.c. Same model memory layout, same int8
// weights (see quant.py), bit-identical to int_forward_ann():
//
//   acc   = b + sum_j x[j] * W[j][:]          x: layer inputs (see below)
//   hidden: h = min(255, (relu(acc) * m_q + 2^23) >> 24)   (uint8)
//   output: logits = acc (int32) -> RESULT_BASE, prediction = first argmax
//
// Inputs of the first layer: uint8 (magic "ANN1", MNIST pixels) or int8
// ("ANN2", e.g. z-scored ECG features); hidden activations are uint8.
// mac_layer() is also the first layer of the direct-encoded SNN ("SNN3"):
// its constant input current is one int8 MAC pass.
//
// Two builds (FW_FAST, see Makefile), identical results:
//   baseline  textbook dense layer: every input x every output, one byte
//             load + multiply + accumulate per synapse (zeros included).
//   fast      the SNN's optimisations, translated to the ANN:
//     * sparsity: only nonzero inputs / activations are processed (the
//       ANN's analogue of event-driven spikes; ~19 % of MNIST pixels are
//       nonzero, ReLU zeroes part of the hidden layer).
//     * SWAR: one 32-bit load fetches 4 int8 weights (offset binary, w+128,
//       converted at boot by model_prepare); even/odd bytes are masked into
//       two 16-bit lanes and ONE multiply scales both lanes by |x|
//       (255 * 255 < 2^16, so the lanes never carry); 8 outputs per pass
//       are kept in registers (16 would spill to the stack, and every stack
//       load costs a full BRAM read here). Negative int8 inputs go to a
//       second list whose products are subtracted. The +128 offset is
//       removed once per output as 128 * sum(x).
//   mac       (FW_MAC, implies FW_FAST) the fast build with the inner loop
//             on the ANN MAC coprocessor (rtl/mac_pcpi.v, custom-1
//             instructions, needs spike_top_both.bit): per nonzero input
//             mac.setx x, then one mac.acc4 per 4 weights of its row; bias
//             via mac.setacc, results via mac.get. The unit removes the
//             +128 weight offset itself (f3[0] = 1). Layers wider than the
//             unit (w_stride > 256) fall back to the fast kernel.

#include "board.h"
#include "hal.h"
#include "snn_layout.h"

#ifndef FW_FAST
#define FW_FAST 0
#endif
#ifndef FW_MAC
#define FW_MAC 0
#endif
#if FW_MAC && !FW_FAST
#error "FW_MAC needs FW_FAST (offset-binary weights)"
#endif

typedef int8_t  s8;
typedef int64_t s64;

static u8  act[2][MAX_NEURONS];             // layer inputs (zero padded)
static s32 acc[MAX_NEURONS];
static volatile s32 *const logits = (volatile s32 *)RESULT_BASE;

#if !FW_FAST
// ------------------------------------------------------------------
// baseline: dense, byte by byte
// ------------------------------------------------------------------
void mac_layer(const layer_desc_t *d, const u8 *x, int x_signed, s32 *out)
{
    const s8  *W    = (const s8 *)(MODEL_BASE + d->w_off);
    const s32 *bias = (const s32 *)(MODEL_BASE + d->b_off);
    // locals: out may alias *d as far as the compiler knows, which would
    // reload n_out from memory on every synapse
    const u32 ni = d->n_in, no = d->n_out, stride = d->w_stride;
    for (u32 o = 0; o < no; o++)
        out[o] = bias[o];
    for (u32 j = 0; j < ni; j++) {
        const s32 xj = x_signed ? (s32)(s8)x[j] : (s32)x[j];
        const s8 *row = W + j * stride;
        for (u32 o = 0; o < no; o++)
            out[o] += xj * row[o];
    }
}

#else
// ------------------------------------------------------------------
// fast: sparse inputs + SWAR multiply-accumulate
// ------------------------------------------------------------------
#define M8 0x00FF00FFu

// nonzero inputs: positive ones from the front [0, n), negative ones
// (int8 inputs only) from the back [m, MAX_NEURONS); xval holds |x|
static const u32 *rowp[MAX_NEURONS];
static u32 xval[MAX_NEURONS];

#if FW_MAC
static void mac_layer_sw(const layer_desc_t *d, const u8 *x, int x_signed,
                         s32 *out)
#else
void mac_layer(const layer_desc_t *d, const u8 *x, int x_signed, s32 *out)
#endif
{
    const u8  *W    = (const u8 *)(MODEL_BASE + d->w_off);
    const s32 *bias = (const s32 *)(MODEL_BASE + d->b_off);
    const u32 stride = d->w_stride;

    // nonzero inputs -> (row pointer, |value|); zero words are skipped whole
    const u32 *xw = (const u32 *)x;
    u32 n = 0, m = MAX_NEURONS;
    s32 sx = 0;
    for (u32 w = 0; w < (d->n_in + 3) / 4; w++) {
        u32 v = xw[w];
        for (u32 j = 4 * w; v; j++, v >>= 8) {
            u32 b = v & 0xFFu;
            if (b) {
                const u32 *r = (const u32 *)(W + j * stride);
                if (x_signed && (b & 0x80u)) {
                    b = 256u - b;
                    rowp[--m] = r;
                    xval[m] = b;
                    sx -= (s32)b;
                } else {
                    rowp[n] = r;
                    xval[n++] = b;
                    sx += (s32)b;
                }
            }
        }
    }
    const s32 corr = 128 * sx;               // undo the +128 weight offset

    for (u32 boff = 0; boff < d->n_out; boff += 8) {    // 8 outputs per pass
        u32 a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        u32 w, e, o;
#define MAC4(OP, k, ae0, ao0, ae1, ao1)                                 \
            w = p[k];                                                   \
            e = (w & M8) * v;          /* outputs 4k (lo), 4k+2 (hi) */ \
            o = ((w >> 8) & M8) * v;   /* outputs 4k+1, 4k+3        */ \
            ae0 OP e & 0xFFFFu; ae1 OP e >> 16;                         \
            ao0 OP o & 0xFFFFu; ao1 OP o >> 16;
        for (u32 i = 0; i < n; i++) {
            const u32 *p = (const u32 *)((const u8 *)rowp[i] + boff);
            const u32 v = xval[i];
            MAC4(+=, 0, a0, a1, a2, a3)
            MAC4(+=, 1, a4, a5, a6, a7)
        }
        for (u32 i = m; i < MAX_NEURONS; i++) {          // negative inputs
            const u32 *p = (const u32 *)((const u8 *)rowp[i] + boff);
            const u32 v = xval[i];
            MAC4(-=, 0, a0, a1, a2, a3)
            MAC4(-=, 1, a4, a5, a6, a7)
        }
#undef MAC4
        const u32 a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
        for (u32 k = 0; k < 8 && boff + k < d->n_out; k++)
            out[boff + k] = bias[boff + k] + (s32)a[k] - corr;
    }
}

#if FW_MAC
// ------------------------------------------------------------------
// mac: the fast kernel's sparse input scan, the multiply-accumulate in the
// MAC coprocessor (accumulator o = output o)
// ------------------------------------------------------------------
#define MACI(f7, f3, a, b) ({ u32 r_;                                    \
        __asm__ volatile(".insn r 0x2B, %3, %4, %0, %1, %2"              \
                         : "=r"(r_) : "r"(a), "r"(b), "i"(f3), "i"(f7));  \
        r_; })
#define MAC_SETX(x)       ((void)MACI(1, 0, x, 0))
#define MAC_ACC4(n, w)    ((void)MACI(2, 1, n, w))  // offset binary weights
#define MAC_GET(n)        ((s32)MACI(3, 0, n, 0))
#define MAC_SETACC(n, v)  ((void)MACI(4, 0, n, v))
#define MAC_NEURONS       256                       // accumulators in mac_pcpi.v

void mac_layer(const layer_desc_t *d, const u8 *x, int x_signed, s32 *out)
{
    const u32 stride = d->w_stride, no = d->n_out;
    if (stride > MAC_NEURONS) {
        mac_layer_sw(d, x, x_signed, out);
        return;
    }
    const u8  *W    = (const u8 *)(MODEL_BASE + d->w_off);
    const s32 *bias = (const s32 *)(MODEL_BASE + d->b_off);

    // accumulators start at the bias (padding outputs no..stride-1 are
    // accumulated too but never read)
    for (u32 o = 0; o < no; o++)
        MAC_SETACC(o, bias[o]);
    // nonzero inputs only; zero words are skipped whole
    const u32 *xw = (const u32 *)x;
    for (u32 w = 0; w < (d->n_in + 3) / 4; w++) {
        u32 v = xw[w];
        for (u32 j = 4 * w; v; j++, v >>= 8) {
            const u32 b = v & 0xFFu;
            if (b) {
                // the unit takes the low 9 bits, signed: int8 or uint8 as is
                MAC_SETX(x_signed ? (u32)(s32)(s8)b : b);
                const u32 *p = (const u32 *)(W + j * stride), *e = p + stride / 4;
                for (u32 nb = 0; p < e; p++, nb += 4)
                    MAC_ACC4(nb, *p);
            }
        }
    }
    for (u32 o = 0; o < no; o++)
        out[o] = MAC_GET(o);
}
#endif
#endif

// ------------------------------------------------------------------
// one inference: input at INPUT_BASE -> logits at RESULT_BASE
// ------------------------------------------------------------------
u32 ann_run(u32 *n_nonzero_hidden)
{
    const u32 L = model->n_layers;
    const int in_signed = model->magic == ANN_S8_MAGIC;
    u8 *x = act[0], *y = act[1];
    u32 nz = 0;

    const volatile u32 *in = (const volatile u32 *)INPUT_BASE;
    for (u32 w = 0; w < (model->n_in + 3) / 4; w++)    // host zero-pads
        ((u32 *)x)[w] = in[w];

    for (u32 l = 0; l < L; l++) {
        const layer_desc_t *d = &model->layer[l];
        mac_layer(d, x, l == 0 && in_signed, acc);
        if (l == L - 1)
            break;
        // ReLU + requantize to uint8 for the next layer
        for (u32 o = 0; o < d->n_out; o++) {
            u32 q = 0;
            if (acc[o] > 0) {
                q = (u32)(((s64)acc[o] * d->p0 + (1 << (ANN_REQ_SHIFT - 1)))
                          >> ANN_REQ_SHIFT);
                if (q > 255)
                    q = 255;
            }
            y[o] = (u8)q;
            nz += q != 0;
        }
        for (u32 o = d->n_out; o < ((d->n_out + 3) & ~3u); o++)
            y[o] = 0;
        u8 *t = x; x = y; y = t;
    }
    *n_nonzero_hidden = nz;

    const u32 n_out = model->layer[L - 1].n_out;
    u32 best = 0;
    for (u32 o = 0; o < n_out; o++) {
        logits[o] = acc[o];
        if (acc[o] > acc[best])
            best = o;
    }
    return best;
}
