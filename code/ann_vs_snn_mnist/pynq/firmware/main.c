// main.c — SNN inference on PicoRV32 (plain RV32IM, no custom
// instructions, unchanged bitstream), used for MNIST and MIT-BIH ECG. The
// ANN kernels are in ann.c; the model type is taken from the magic of
// model.bin (snn_layout.h): SNN2 rate-coded spike inputs, SNN3 int8 inputs
// with direct encoding, ANN1 uint8 / ANN2 int8 inputs.
//
// The PS writes model.bin (quant.py) to MODEL_BASE before releasing the CPU
// reset. Per sample it writes the input (SNN2: spike bitmaps, otherwise the
// input bytes) to INPUT_BASE and sends MB_CMD_RUN; the SNN runs T timesteps
// of the LIF layers, writes the output spike counts to RESULT_BASE and
// replies with the predicted class (first argmax), the elapsed timer ticks
// and the number of spikes of all layers. With direct encoding the first
// layer's current b + W x is computed once (mac_layer in ann.c, counted in
// the timed region) and reused every timestep.
//
// Integer LIF, identical to int_forward_snn() in quant.py:
//   mem = ((beta_q16 * mem) >> 16) + I - (spiked last step ? thr : 0)
//   spike = mem > thr
//
// Two builds of this file (see Makefile), bit-identical results:
//   snn_mnist.bin       FW_FAST=0  baseline: per spike, add one weight row
//                                   to the currents byte by byte.
//   snn_mnist_fast.bin  FW_FAST=1  same maths, restructured:
//     * delta currents: I_t = I_{t-1} + sum W[inputs that turned on]
//                                     - sum W[inputs that turned off].
//       Rate-coded MNIST pixels that are saturated spike every step, so
//       only ~31 of ~95 active inputs change per step (exact, integers).
//     * SWAR accumulation: one 32-bit load fetches 4 int8 weights; even and
//       odd bytes are added into two 16-bit lanes per register
//       (offset-binary, w+128, flushed every <=256 rows), 16 outputs per
//       pass kept in 8 registers -> ~1.5 instructions per synapse instead
//       of 7.
//     * spike lists via de Bruijn count-trailing-zeros instead of a bit loop.
// snn_mnist_muriscv.bin (FW_FAST=1, FW_MURISCV=1) is the fast image plus
// "ANN3" models, run by the unmodified muRISCV-NN library (external ANN
// reference, ../muriscv/ann_muriscv.c); all other models as in the fast image.
// snn_mnist_mac.bin (FW_HW=1, FW_MAC=1) is the hw image with ann.c's MAC
// layer on the ANN MAC coprocessor (spike_top_both.bit): ANN / GRU on the
// MAC unit, SNN on the SNN unit, the direct-encoded SNN's first layer (a MAC
// pass) on the MAC unit. FW_MAC only changes the boot banner of this file.

#include "board.h"
#include "hal.h"
#include "snn_layout.h"

#ifndef FW_FAST
#define FW_FAST 0
#endif
#ifndef FW_HW                   // SNN coprocessor kernel (implies FW_FAST)
#define FW_HW 0
#endif
#ifndef FW_MAC                  // ann.c on the MAC coprocessor (with FW_HW)
#define FW_MAC 0
#endif
#ifndef FW_MURISCV              // + ANN3 models on muRISCV-NN (implies FW_FAST)
#define FW_MURISCV 0
#endif
#if FW_MURISCV
#include "ann_muriscv.h"        // glue to the unmodified library, ../muriscv/
#endif

typedef int8_t  s8;
typedef int64_t s64;

// ------------------------------------------------------------------
// console ring (in BRAM, drained by the PS)
// ------------------------------------------------------------------
static console_ring_t *const con = (console_ring_t *)CONSOLE_BASE;

static void con_putc(char c)
{
    if (c == '\n')
        con_putc('\r');
    u32 h = con->head, t = con->tail;
    if (h - t >= 512)
        return;                        // full: drop
    con->buf[h & CONSOLE_BUF_MASK] = c;
    con->head = h + 1;
}

static void con_puts(const char *s)
{
    while (*s)
        con_putc(*s++);
}

static void con_putdec(u32 v)
{
    char b[12];
    int i = 0;
    do {
        b[i++] = '0' + (v % 10);
        v /= 10;
    } while (v);
    while (i)
        con_putc(b[--i]);
}

// ------------------------------------------------------------------
// model (in BRAM at MODEL_BASE, written by the PS; header in snn_layout.h)
// ------------------------------------------------------------------
static inline int model_is_snn(void)
{
    return model->magic == SNN_MAGIC || model->magic == SNN_DIRECT_MAGIC;
}

// RSNN ("RSN1", streaming ECG, quant.py): layer 0 is a recurrent LIF layer
// whose input at step t is [n_ev input events of step t | its own H spikes
// of step t-1]; layers 1 and 2 are a linear readout of the hidden spike
// counts and of 3 int8 RR features, evaluated once after the last step:
//   logits = (W1^T counts) * m1 + (W2^T rr) * m2 + b1
static u32 rsnn_check(void)
{
    const layer_desc_t *d0 = &model->layer[0], *d1 = &model->layer[1],
                       *d2 = &model->layer[2];
    if (model->n_layers != 3 || d0->n_out == 0 || d0->n_in <= d0->n_out ||
        d0->n_in - d0->n_out > 32 || d0->n_out > RSNN_MAX_H || d0->p1 <= 0 ||
        d0->w_stride > SNN_MAX_OUT ||
        d2->p1 < 1 || d2->p1 > GRU_BINS_MAX ||
        d1->n_in != (u32)d2->p1 * d0->n_out || d1->n_out != d2->n_out || d2->n_in > 4 ||
        d1->n_out > MAX_OUT)
        return SNN_ERR_MODEL;
    for (u32 l = 0; l < 3; l++) {
        const layer_desc_t *d = &model->layer[l];
        if (d->n_out > d->w_stride || d->w_stride > MAX_NEURONS ||
            (d->w_stride & 15) || ((d->w_off | d->b_off) & 3) ||
            d->w_off + d->n_in * d->w_stride > MODEL_MAX_BYTES ||
            d->b_off + 4 * d->n_out > MODEL_MAX_BYTES)
            return SNN_ERR_MODEL;
    }
#if FW_HW
    if (d0->w_stride > SNN_COP_NEURONS)
        return SNN_ERR_MODEL;
#endif
    return SNN_OK;
}

#if FW_MURISCV
// "ANN3" (muRISCV-NN / CMSIS-NN layout, ann_muriscv.c) shares only the first
// 4 header words with model_hdr_t: magic, n_layers, n_in, 0
static u32 ann3_n_out(void)
{
    const u32 *h = (const u32 *)MODEL_BASE;
    return h[4 + 7 * (h[1] - 1) + 1];          // n_out of the last layer
}

static u32 ann3_in[MAX_NEURONS / 4];       // int8 input, word aligned
static s8 ann3_act[2][ANN3_MAX_WIDTH];     // hidden activations

// one inference with the library; copies the input like ann_run (ann.c)
static u32 ann3_fw_run(u32 *n_nonzero_hidden)
{
    const volatile u32 *in = (const volatile u32 *)INPUT_BASE;
    for (u32 w = 0; w < (model->n_in + 3) / 4; w++)    // host zero-pads
        ann3_in[w] = in[w];
    return (u32)ann3_run((const uint8_t *)MODEL_BASE, (const int8_t *)ann3_in,
                         ann3_act[0], ann3_act[1], (int32_t *)RESULT_BASE,
                         n_nonzero_hidden);
}
#endif

static u32 model_check(void)
{
#if FW_MURISCV
    if (model->magic == ANN3_MAGIC)
        return ann3_check((const uint8_t *)MODEL_BASE, MODEL_MAX_BYTES) ||
                       model->n_in > MAX_NEURONS || ann3_n_out() > MAX_OUT
                   ? SNN_ERR_MODEL : SNN_OK;
#endif
    if (model->magic == RSNN_MAGIC)
        return rsnn_check();
    if (model->magic == GRU_MAGIC)
        return gru_check();
    const int is_snn = model_is_snn();
    if ((!is_snn && model->magic != ANN_MAGIC && model->magic != ANN_S8_MAGIC) ||
        model->n_layers == 0 ||
        model->n_layers > MAX_LAYERS || model->n_in > MAX_NEURONS)
        return SNN_ERR_MODEL;
    u32 n_prev = model->n_in;
    for (u32 l = 0; l < model->n_layers; l++) {
        const layer_desc_t *d = &model->layer[l];
        if (d->n_in != n_prev || d->n_out == 0 || d->n_out > d->w_stride ||
            d->w_stride > MAX_NEURONS || (d->w_stride & 15) ||
            (is_snn && d->p1 <= 0) ||                          // SNN: thr > 0
            (is_snn && d->w_stride > SNN_MAX_OUT) ||           // SNN state size
            (!is_snn && (d->p0 <= 0) != (l == model->n_layers - 1)) || // ANN m_q
            d->w_off + d->n_in * d->w_stride > MODEL_MAX_BYTES ||
            d->b_off + 4 * d->n_out > MODEL_MAX_BYTES ||
            ((d->w_off | d->b_off) & 3))
            return SNN_ERR_MODEL;
        n_prev = d->n_out;
    }
#if FW_HW
    if (is_snn) {                          // neurons in the SNN coprocessor
        u32 nb = 0;
        for (u32 l = 0; l < model->n_layers; l++)
            nb += model->layer[l].w_stride;
        if (nb > SNN_COP_NEURONS || model->n_layers > 4)
            return SNN_ERR_MODEL;
    }
#endif
    return n_prev <= MAX_OUT ? SNN_OK : SNN_ERR_MODEL;
}

// ------------------------------------------------------------------
// SNN state
// ------------------------------------------------------------------
#if !FW_HW
static s32 mem[MAX_LAYERS][SNN_MAX_OUT];   // membrane potentials
#endif
static volatile u32 *const out_counts = (volatile u32 *)RESULT_BASE;

// direct encoding ("SNN3"): the int8 input is a constant current, so the
// first layer's current I = b + W x is one MAC pass (ann.c), reused every
// timestep; the layers after it are spike-driven as for "SNN2".
static u32 x_in[MAX_NEURONS / 4];          // int8 input, word aligned

static void direct_current(s32 *out)
{
    const volatile u32 *in = (const volatile u32 *)INPUT_BASE;
    for (u32 w = 0; w < (model->n_in + 3) / 4; w++)    // host zero-pads
        x_in[w] = in[w];
    mac_layer(&model->layer[0], (const u8 *)x_in, 1, out);
}

// RSNN readout (all builds): counts_h -> logits at RESULT_BASE, first argmax.
// Fast / hw builds keep the weights in offset binary (model_prepare).
static u16 counts_h[GRU_BINS_MAX * RSNN_MAX_H];  // spike counts [bin][neuron]
#if FW_FAST
#define WVAL(x) ((s32)(x) - 128)
#else
#define WVAL(x) ((s32)(s8)(x))
#endif

static u32 rsnn_readout(u32 T)
{
    const layer_desc_t *d1 = &model->layer[1], *d2 = &model->layer[2];
    const u8 *W1 = (const u8 *)(MODEL_BASE + d1->w_off);
    const u8 *W2 = (const u8 *)(MODEL_BASE + d2->w_off);
    const s32 *b = (const s32 *)(MODEL_BASE + d1->b_off);
    const volatile s8 *rr = (const volatile s8 *)(INPUT_BASE + 4 * T);
    volatile s32 *logits = (volatile s32 *)RESULT_BASE;
    u32 best = 0;
    s32 best_v = 0;
    for (u32 c = 0; c < d1->n_out; c++) {
        s32 a1 = 0, a2 = 0;
        for (u32 o = 0; o < d1->n_in; o++)
            a1 += (s32)counts_h[o] * WVAL(W1[o * d1->w_stride + c]);
        for (u32 j = 0; j < d2->n_in; j++)
            a2 += (s32)rr[j] * WVAL(W2[j * d2->w_stride + c]);
        s32 v = a1 * d1->p0 + a2 * d2->p0 + b[c];
        logits[c] = v;
        if (c == 0 || v > best_v) {
            best = c;
            best_v = v;
        }
    }
    return best;
}

// mem_prev > thr  <=>  the neuron spiked in the previous step (thr > 0,
// mem starts at 0), so the reset needs no separate spike array.
static inline s32 lif(const layer_desc_t *d, s32 *m, s32 i)
{
    s32 mo = *m;
    s32 v = (s32)(((s64)d->p0 * mo) >> 16) + i;       // p0 = beta_q16
    if (mo > d->p1)                                   // p1 = thr
        v -= d->p1;
    *m = v;
    return v > d->p1;
}

#if !FW_FAST
// ------------------------------------------------------------------
// baseline kernel
// ------------------------------------------------------------------
static u16 act[2][MAX_NEURONS];            // indices of active inputs
static s32 cur[MAX_NEURONS];               // input current of one layer
static s32 cur0[MAX_NEURONS];              // direct encoding: layer 0 current

static u32 run_sample(u32 T, u32 *n_spikes)
{
    const u32 L = model->n_layers;
    const u32 n_out = model->layer[L - 1].n_out;
    const u32 wps = (model->n_in + 31) / 32;   // input words per timestep
    const int direct = model->magic == SNN_DIRECT_MAGIC;

    for (u32 l = 0; l < L; l++)
        for (u32 o = 0; o < model->layer[l].n_out; o++)
            mem[l][o] = 0;
    for (u32 o = 0; o < n_out; o++)
        out_counts[o] = 0;
    *n_spikes = 0;
    if (direct)
        direct_current(cur0);                  // layer 0 current, once

    for (u32 t = 0; t < T; t++) {
        // input bitmap -> list of active input indices
        u16 *a_in = act[0], *a_out = act[1];
        u32 n_act = 0;
        if (!direct) {
            const volatile u32 *in = (const volatile u32 *)INPUT_BASE + t * wps;
            for (u32 w = 0; w < wps; w++) {
                u32 bits = in[w];
                for (u32 b = 0; bits; b++, bits >>= 1)
                    if (bits & 1u)
                        a_in[n_act++] = (u16)(32 * w + b);
            }
        }

        for (u32 l = 0; l < L; l++) {
            const layer_desc_t *d = &model->layer[l];
            const s8  *W    = (const s8 *)(MODEL_BASE + d->w_off);
            const s32 *bias = (const s32 *)(MODEL_BASE + d->b_off);
            const u32 no = d->n_out;
            const s32 *c = cur;

            if (l == 0 && direct) {
                c = cur0;                      // constant input current
            } else {
                // I = b + sum of the weight rows of the active inputs
                for (u32 o = 0; o < no; o++)
                    cur[o] = bias[o];
                for (u32 i = 0; i < n_act; i++) {
                    const s8 *row = W + (u32)a_in[i] * d->w_stride;
                    for (u32 o = 0; o < no; o++)
                        cur[o] += row[o];
                }
            }

            // LIF update, collect this layer's spikes as next layer input
            u32 n_next = 0;
            for (u32 o = 0; o < no; o++)
                if (lif(d, &mem[l][o], c[o]))
                    a_out[n_next++] = (u16)o;
            *n_spikes += n_next;

            u16 *tmp = a_in; a_in = a_out; a_out = tmp;
            n_act = n_next;
        }

        for (u32 i = 0; i < n_act; i++)        // output-layer spikes
            out_counts[a_in[i]]++;
    }

    u32 best = 0;
    for (u32 o = 1; o < n_out; o++)
        if (out_counts[o] > out_counts[best])
            best = o;
    return best;
}

// RSNN, baseline: per step, the current from scratch (active input events +
// last step's hidden spikes, one weight row each, byte by byte), then LIF
static u32 run_rsnn(u32 T, u32 *n_spikes)
{
    const layer_desc_t *d = &model->layer[0];
    const u32 H = d->n_out, n_ev = d->n_in - H;
    const s8 *W = (const s8 *)(MODEL_BASE + d->w_off);
    const volatile u32 *in = (const volatile u32 *)INPUT_BASE;
    u16 *a = act[0], *prev = act[1];       // active inputs, spikes of t-1
    u32 n_prev = 0, spikes = 0;

    const u32 bins = (u32)model->layer[2].p1;
    for (u32 o = 0; o < H; o++)
        mem[0][o] = 0;
    for (u32 o = 0; o < bins * H; o++)
        counts_h[o] = 0;
    for (u32 t = 0; t < T; t++) {
        u16 *wcnt = counts_h + (t * bins / T) * H;  // window of step t
        u32 n = 0;
        u32 bits = in[t];
        for (u32 b = 0; bits; b++, bits >>= 1)
            if (bits & 1u)
                a[n++] = (u16)b;
        for (u32 i = 0; i < n_prev; i++)
            a[n++] = (u16)(n_ev + prev[i]);
        for (u32 o = 0; o < H; o++)
            cur[o] = 0;                        // no bias
        for (u32 i = 0; i < n; i++) {
            const s8 *row = W + (u32)a[i] * d->w_stride;
            for (u32 o = 0; o < H; o++)
                cur[o] += row[o];
        }
        n_prev = 0;
        for (u32 o = 0; o < H; o++)
            if (lif(d, &mem[0][o], cur[o])) {
                prev[n_prev++] = (u16)o;
                wcnt[o]++;
            }
        spikes += n_prev;
    }
    *n_spikes = spikes;
    return rsnn_readout(T);
}

static void model_prepare(void) {}

#else
// ------------------------------------------------------------------
// fast kernel (delta currents + SWAR, see top of file)
// ------------------------------------------------------------------
#define NW_MAX  (MAX_NEURONS / 32)
#define M8      0x00FF00FFu             // even bytes -> two 16-bit lanes
#define CHUNK   256                     // 255 * 257 < 65536: no lane overflow

#if !FW_HW
static s32 dcur[MAX_LAYERS][SNN_MAX_OUT];  // persistent currents I_t
#endif
static u32 prev_in[MAX_LAYERS][NW_MAX];    // input bitmap of the last step
static u32 bits_buf[2][NW_MAX];            // input / output bitmap of a layer
static u16 lst_on[MAX_NEURONS], lst_off[MAX_NEURONS];
#if !FW_HW
static const u32 *rowp[MAX_NEURONS];
#endif

static const u8 debruijn_ctz[32] = {
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9,
};

// set bits of x -> indices base + bit, appended to lst
static inline u32 bits_to_list(u32 x, u32 base, u16 *lst, u32 n)
{
    while (x) {
        u32 low = x & -x;
        lst[n++] = (u16)(base + debruijn_ctz[(low * 0x077CB531u) >> 27]);
        x ^= low;
    }
    return n;
}

#if !FW_HW
// c[o] += sign * sum over rows in lst of W[row][o], for all w_stride outputs.
// The weights were converted to offset binary (w + 128, unsigned) at boot.
static void accumulate(s32 *c, const u8 *W, u32 stride,
                       const u16 *lst, u32 n, int negate)
{
    if (n == 0)
        return;
    for (u32 i = 0; i < n; i++)
        rowp[i] = (const u32 *)(W + (u32)lst[i] * stride);

#define LANES(a, k)                                                     \
    do {                                                                \
        s32 lo = (s32)((a) & 0xFFFFu) - off, hi = (s32)((a) >> 16) - off; \
        if (negate) { cb[k] -= lo; cb[(k) + 2] -= hi; }                 \
        else        { cb[k] += lo; cb[(k) + 2] += hi; }                 \
    } while (0)

    for (u32 boff = 0; boff < stride; boff += 16) {      // 16 outputs per pass
        s32 *cb = c + boff;
        for (u32 s = 0; s < n; s += CHUNK) {
            u32 e = n - s > CHUNK ? s + CHUNK : n;
            u32 a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
            for (u32 i = s; i < e; i++) {
                const u32 *p = (const u32 *)((const u8 *)rowp[i] + boff);
                u32 w;
                w = p[0]; a0 += w & M8; a1 += (w >> 8) & M8;
                w = p[1]; a2 += w & M8; a3 += (w >> 8) & M8;
                w = p[2]; a4 += w & M8; a5 += (w >> 8) & M8;
                w = p[3]; a6 += w & M8; a7 += (w >> 8) & M8;
            }
            // word j holds outputs 4j..4j+3 (little endian): the even-byte
            // register has 4j (lo) / 4j+2 (hi), the odd one 4j+1 / 4j+3
            const s32 off = 128 * (s32)(e - s);
            LANES(a0, 0);  LANES(a1, 1);
            LANES(a2, 4);  LANES(a3, 5);
            LANES(a4, 8);  LANES(a5, 9);
            LANES(a6, 12); LANES(a7, 13);
        }
    }
#undef LANES
}

#endif

// u_t = [input events of step t | hidden spikes of t-1], as a bitmap
static u32 hbits[NW_MAX];                  // hidden spikes of the last step

static void rsnn_input(u32 *bin, u32 word, u32 n_ev, u32 H)
{
    const u32 nw = (n_ev + H + 31) / 32, sh = n_ev & 31, base = n_ev >> 5;
    for (u32 w = 0; w < nw; w++)
        bin[w] = 0;
    bin[0] = word;
    for (u32 w = 0; w < (H + 31) / 32; w++) {
        u32 x = hbits[w];
        bin[base + w] |= x << sh;
        if (sh && base + w + 1 < nw)
            bin[base + w + 1] |= x >> (32 - sh);
    }
}

// int8 -> offset binary (w + 128) in place, once after the model is loaded
static void model_prepare(void)
{
    for (u32 l = 0; l < model->n_layers; l++) {
        const layer_desc_t *d = &model->layer[l];
        volatile u32 *w = (volatile u32 *)(MODEL_BASE + d->w_off);
        for (u32 i = 0; i < d->n_in * d->w_stride / 4; i++)
            w[i] ^= 0x80808080u;
    }
}

#if !FW_HW
static u32 run_sample(u32 T, u32 *n_spikes)
{
    const u32 L = model->n_layers;
    const u32 n_out = model->layer[L - 1].n_out;
    const u32 wps = (model->n_in + 31) / 32;
    const int direct = model->magic == SNN_DIRECT_MAGIC;

    for (u32 l = 0; l < L; l++) {
        const layer_desc_t *d = &model->layer[l];
        const s32 *bias = (const s32 *)(MODEL_BASE + d->b_off);
        for (u32 o = 0; o < d->w_stride; o++) {
            mem[l][o] = 0;
            dcur[l][o] = o < d->n_out ? bias[o] : 0;
        }
        for (u32 w = 0; w < (d->n_in + 31) / 32; w++)
            prev_in[l][w] = 0;
    }
    if (direct)                                // layer 0 current, once
        direct_current(dcur[0]);
    for (u32 o = 0; o < n_out; o++)
        out_counts[o] = 0;
    u32 spikes = 0;

    for (u32 t = 0; t < T; t++) {
        u32 *bin = bits_buf[0], *bout = bits_buf[1];
        if (!direct) {
            const volatile u32 *in = (const volatile u32 *)INPUT_BASE + t * wps;
            for (u32 w = 0; w < wps; w++)
                bin[w] = in[w];
        }

        for (u32 l = 0; l < L; l++) {
            const layer_desc_t *d = &model->layer[l];
            const u8 *W = (const u8 *)(MODEL_BASE + d->w_off);
            const u32 no = d->n_out;

            // inputs that changed since the last step (direct encoding:
            // the first layer's current is constant, nothing to update)
            if (l > 0 || !direct) {
                u32 n_on = 0, n_off = 0;
                for (u32 w = 0; w < (d->n_in + 31) / 32; w++) {
                    u32 x = bin[w], p = prev_in[l][w], ch = x ^ p;
                    prev_in[l][w] = x;
                    n_on  = bits_to_list(ch & x, 32 * w, lst_on, n_on);
                    n_off = bits_to_list(ch & p, 32 * w, lst_off, n_off);
                }
                accumulate(dcur[l], W, d->w_stride, lst_on, n_on, 0);
                accumulate(dcur[l], W, d->w_stride, lst_off, n_off, 1);
            }

            // LIF update -> output bitmap (input of the next layer)
            for (u32 w = 0; w < (no + 31) / 32; w++)
                bout[w] = 0;
            for (u32 o = 0; o < no; o++)
                if (lif(d, &mem[l][o], dcur[l][o])) {
                    bout[o >> 5] |= 1u << (o & 31);
                    spikes++;
                    if (l == L - 1)
                        out_counts[o]++;
                }

            u32 *tmp = bin; bin = bout; bout = tmp;
        }
    }
    *n_spikes = spikes;

    u32 best = 0;
    for (u32 o = 1; o < n_out; o++)
        if (out_counts[o] > out_counts[best])
            best = o;
    return best;
}

// RSNN, fast: delta currents (rows of the inputs and hidden neurons whose
// state changed, SWAR) + LIF in software
static u32 run_rsnn(u32 T, u32 *n_spikes)
{
    const layer_desc_t *d = &model->layer[0];
    const u32 H = d->n_out, n_ev = d->n_in - H, nw = (d->n_in + 31) / 32;
    const u8 *W = (const u8 *)(MODEL_BASE + d->w_off);
    const volatile u32 *in = (const volatile u32 *)INPUT_BASE;
    u32 *bin = bits_buf[0];
    u32 spikes = 0;

    for (u32 o = 0; o < d->w_stride; o++) {
        mem[0][o] = 0;
        dcur[0][o] = 0;                        // no bias
    }
    const u32 bins = (u32)model->layer[2].p1;
    for (u32 o = 0; o < bins * H; o++)
        counts_h[o] = 0;
    for (u32 w = 0; w < NW_MAX; w++)
        prev_in[0][w] = hbits[w] = 0;

    for (u32 t = 0; t < T; t++) {
        u16 *wcnt = counts_h + (t * bins / T) * H;  // window of step t
        rsnn_input(bin, in[t], n_ev, H);
        u32 n_on = 0, n_off = 0;
        for (u32 w = 0; w < nw; w++) {
            u32 x = bin[w], p = prev_in[0][w], ch = x ^ p;
            prev_in[0][w] = x;
            n_on  = bits_to_list(ch & x, 32 * w, lst_on, n_on);
            n_off = bits_to_list(ch & p, 32 * w, lst_off, n_off);
        }
        accumulate(dcur[0], W, d->w_stride, lst_on, n_on, 0);
        accumulate(dcur[0], W, d->w_stride, lst_off, n_off, 1);
        for (u32 w = 0; w < (H + 31) / 32; w++)
            hbits[w] = 0;
        for (u32 o = 0; o < H; o++)
            if (lif(d, &mem[0][o], dcur[0][o])) {
                hbits[o >> 5] |= 1u << (o & 31);
                wcnt[o]++;
                spikes++;
            }
    }
    *n_spikes = spikes;
    return rsnn_readout(T);
}
#else
// ------------------------------------------------------------------
// hw kernel (FW_HW): the fast kernel's structure (delta inputs, spike
// bitmaps), with the neuron state, the current accumulation and the LIF
// update in the SNN coprocessor (rtl/snn_pcpi.v, custom-0 instructions).
// Needs spike_top_snn.bit. Layer l's neurons are coprocessor neurons
// nbase[l] .. nbase[l] + w_stride - 1 (padding neurons get zero weights).
// ------------------------------------------------------------------
#define COP(f7, f3, a, b) ({ u32 r_;                                     \
        __asm__ volatile(".insn r 0x0B, %3, %4, %0, %1, %2"              \
                         : "=r"(r_) : "r"(a), "r"(b), "i"(f3), "i"(f7));  \
        r_; })
#define SNN_CFG(slot, beta, thr)  ((void)COP(0, slot, beta, thr))
#define SNN_CLR(first, n)         ((void)COP(1, 0, first, n))
#define SNN_SETCUR(n, v)          ((void)COP(2, 0, n, v))
#define SNN_ADD4(n, w)            ((void)COP(3, 1, n, w))   // offset binary
#define SNN_SUB4(n, w)            ((void)COP(3, 3, n, w))
#define SNN_STEP(slot, first, n)  COP(4, slot, first, n)

static u32 nbase[MAX_LAYERS];              // first coprocessor neuron of a layer
static s32 cur0[MAX_NEURONS];              // direct encoding: layer 0 current

// slot (= layer) is an immediate of the instruction
static inline void snn_cfg(u32 l, s32 beta, s32 thr)
{
    switch (l) {
    case 0:  SNN_CFG(0, beta, thr); break;
    case 1:  SNN_CFG(1, beta, thr); break;
    case 2:  SNN_CFG(2, beta, thr); break;
    default: SNN_CFG(3, beta, thr); break;
    }
}

static inline u32 snn_step(u32 l, u32 first, u32 n)
{
    switch (l) {
    case 0:  return SNN_STEP(0, first, n);
    case 1:  return SNN_STEP(1, first, n);
    case 2:  return SNN_STEP(2, first, n);
    default: return SNN_STEP(3, first, n);
    }
}

// add (sub = 0) or subtract the weight rows of the listed inputs to the
// currents of the neurons base .. base + stride - 1: 4 weights per instruction
static void rows_hw(const u8 *W, u32 stride, u32 base, const u16 *lst, u32 n,
                    int sub)
{
    for (u32 i = 0; i < n; i++) {
        const u32 *row = (const u32 *)(W + (u32)lst[i] * stride);
        u32 nb = base;
        if (sub)
            for (u32 w = 0; w < stride / 4; w++, nb += 4)
                SNN_SUB4(nb, row[w]);
        else
            for (u32 w = 0; w < stride / 4; w++, nb += 4)
                SNN_ADD4(nb, row[w]);
    }
}

static u32 run_sample(u32 T, u32 *n_spikes)
{
    const u32 L = model->n_layers;
    const u32 n_out = model->layer[L - 1].n_out;
    const u32 wps = (model->n_in + 31) / 32;
    const int direct = model->magic == SNN_DIRECT_MAGIC;

    u32 nb = 0;
    for (u32 l = 0; l < L; l++) {
        const layer_desc_t *d = &model->layer[l];
        nbase[l] = nb;
        nb += d->w_stride;
        snn_cfg(l, d->p0, d->p1);
        for (u32 w = 0; w < (d->n_in + 31) / 32; w++)
            prev_in[l][w] = 0;
    }
    SNN_CLR(0, nb);                            // mem = cur = 0
    if (direct)
        direct_current(cur0);                  // layer 0 current, once
    for (u32 l = 0; l < L; l++) {              // currents start at the bias
        const layer_desc_t *d = &model->layer[l];
        const s32 *bias = (l == 0 && direct) ? cur0
                          : (const s32 *)(MODEL_BASE + d->b_off);
        for (u32 o = 0; o < d->n_out; o++)
            SNN_SETCUR(nbase[l] + o, bias[o]);
    }
    for (u32 o = 0; o < n_out; o++)
        out_counts[o] = 0;
    u32 spikes = 0;

    for (u32 t = 0; t < T; t++) {
        u32 *bin = bits_buf[0], *bout = bits_buf[1];
        if (!direct) {
            const volatile u32 *in = (const volatile u32 *)INPUT_BASE + t * wps;
            for (u32 w = 0; w < wps; w++)
                bin[w] = in[w];
        }

        for (u32 l = 0; l < L; l++) {
            const layer_desc_t *d = &model->layer[l];
            const u8 *W = (const u8 *)(MODEL_BASE + d->w_off);
            const u32 no = d->n_out;

            // inputs that changed since the last step -> weight rows
            if (l > 0 || !direct) {
                u32 n_on = 0, n_off = 0;
                for (u32 w = 0; w < (d->n_in + 31) / 32; w++) {
                    u32 x = bin[w], p = prev_in[l][w], ch = x ^ p;
                    prev_in[l][w] = x;
                    n_on  = bits_to_list(ch & x, 32 * w, lst_on, n_on);
                    n_off = bits_to_list(ch & p, 32 * w, lst_off, n_off);
                }
                rows_hw(W, d->w_stride, nbase[l], lst_on, n_on, 0);
                rows_hw(W, d->w_stride, nbase[l], lst_off, n_off, 1);
            }

            // LIF update in the coprocessor, 32 neurons per instruction
            for (u32 g = 0; 32 * g < no; g++) {
                u32 cnt = no - 32 * g < 32 ? no - 32 * g : 32;
                u32 m = snn_step(l, nbase[l] + 32 * g, cnt);
                bout[g] = m;
                for (u32 x = m; x; x &= x - 1)
                    spikes++;
                if (l == L - 1)
                    for (u32 x = m; x; x &= x - 1)
                        out_counts[32 * g + debruijn_ctz[((x & -x) * 0x077CB531u) >> 27]]++;
            }
            u32 *tmp = bin; bin = bout; bout = tmp;
        }
    }
    *n_spikes = spikes;

    u32 best = 0;
    for (u32 o = 1; o < n_out; o++)
        if (out_counts[o] > out_counts[best])
            best = o;
    return best;
}

// RSNN, hw: as fast, with the neuron state, the row additions and the LIF
// update in the SNN coprocessor
static u32 run_rsnn(u32 T, u32 *n_spikes)
{
    const layer_desc_t *d = &model->layer[0];
    const u32 H = d->n_out, n_ev = d->n_in - H, nw = (d->n_in + 31) / 32;
    const u8 *W = (const u8 *)(MODEL_BASE + d->w_off);
    const volatile u32 *in = (const volatile u32 *)INPUT_BASE;
    u32 *bin = bits_buf[0];
    u32 spikes = 0;

    snn_cfg(0, d->p0, d->p1);
    SNN_CLR(0, d->w_stride);                   // mem = cur = 0 (no bias)
    const u32 bins = (u32)model->layer[2].p1;
    for (u32 o = 0; o < bins * H; o++)
        counts_h[o] = 0;
    for (u32 w = 0; w < NW_MAX; w++)
        prev_in[0][w] = hbits[w] = 0;

    for (u32 t = 0; t < T; t++) {
        u16 *wcnt = counts_h + (t * bins / T) * H;  // window of step t
        rsnn_input(bin, in[t], n_ev, H);
        u32 n_on = 0, n_off = 0;
        for (u32 w = 0; w < nw; w++) {
            u32 x = bin[w], p = prev_in[0][w], ch = x ^ p;
            prev_in[0][w] = x;
            n_on  = bits_to_list(ch & x, 32 * w, lst_on, n_on);
            n_off = bits_to_list(ch & p, 32 * w, lst_off, n_off);
        }
        rows_hw(W, d->w_stride, 0, lst_on, n_on, 0);
        rows_hw(W, d->w_stride, 0, lst_off, n_off, 1);
        for (u32 g = 0; 32 * g < H; g++) {
            u32 cnt = H - 32 * g < 32 ? H - 32 * g : 32;
            u32 m = snn_step(0, 32 * g, cnt);
            hbits[g] = m;
            for (u32 x = m; x; x &= x - 1) {
                wcnt[32 * g + debruijn_ctz[((x & -x) * 0x077CB531u) >> 27]]++;
                spikes++;
            }
        }
    }
    *n_spikes = spikes;
    return rsnn_readout(T);
}
#endif
#endif

// ------------------------------------------------------------------
// mailbox (commands from the PS)
// ------------------------------------------------------------------
static mailbox_t *const mb = (mailbox_t *)MAILBOX_BASE;
static u32 model_err;

static void mb_reply(u32 r0, u32 r1, u32 r2, u32 r3)
{
    mb->resp[0] = r0;
    mb->resp[1] = r1;
    mb->resp[2] = r2;
    mb->resp[3] = r3;
    mb->seq_out = mb->seq_in;
}

static void mb_service(void)
{
    if (mb->seq_in == mb->seq_out)
        return;

    u32 c = mb->cmd[0], a0 = mb->cmd[1], a1 = mb->cmd[2], a2 = mb->cmd[3];

    switch (c) {
    case MB_CMD_NOP:
        mb_reply(0, 0, 0, 0);
        break;
    case MB_CMD_ECHO:
        mb_reply(a0, a1, a2, 0);
        break;
    case MB_CMD_RUN: {
        if (model_err) {
            mb_reply(0, 0, 0, SNN_ERR_MODEL);
            break;
        }
        // a0 = T. Spike inputs: T bitmaps must fit the input area; int8 /
        // uint8 inputs (n_in <= MAX_NEURONS bytes) always fit
        const int is_snn = model_is_snn();
        const int is_rsnn = model->magic == RSNN_MAGIC;
        const int is_gru = model->magic == GRU_MAGIC;
        u32 wps = (model->n_in + 31) / 32;
        if ((is_snn && (a0 == 0 || (model->magic == SNN_MAGIC &&
                                    a0 * wps * 4 > INPUT_MAX_BYTES))) ||
            (is_rsnn && (a0 == 0 || 4 * a0 + 4 > INPUT_MAX_BYTES)) ||
            (is_gru && (a0 == 0 || ((a0 + 3) & ~3u) + 4 > INPUT_MAX_BYTES))) {
            mb_reply(0, 0, 0, SNN_ERR_INPUT);
            break;
        }
        led_write(0x002u);
        u32 aux;
        u32 t0 = timer_now();
#if FW_MURISCV
        u32 pred = model->magic == ANN3_MAGIC ? ann3_fw_run(&aux)
                 : is_gru ? gru_run(a0, &aux)
#else
        u32 pred = is_gru ? gru_run(a0, &aux)
#endif
                 : is_rsnn ? run_rsnn(a0, &aux)
                 : is_snn ? run_sample(a0, &aux) : ann_run(&aux);
        u32 dt = timer_now() - t0;
        led_write(0x001u);
        mb_reply(pred, dt, aux, SNN_OK);
        break;
    }
    default:
        mb_reply(0xDEADBEEFu, 0, 0, 0);
        break;
    }
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
void main(void)
{
    con->head = 0;
    con->tail = 0;
    mb->seq_out = 0;
    led_write(0x001u);

#if FW_HW && FW_MAC
    con_puts("ANN/SNN rv firmware v3.3 MAC (fast + SNN and MAC coprocessors, rv32im + custom-0/1)\n");
#elif FW_HW
    con_puts("ANN/SNN rv firmware v3.3 HW (fast + SNN coprocessor, rv32im + custom-0)\n");
#elif FW_MURISCV
    con_puts("ANN/SNN rv firmware v3.3 MURISCV (fast + muRISCV-NN ANN, rv32im)\n");
#else
    con_puts(FW_FAST ? "ANN/SNN rv firmware v3.3 FAST (sparse + SWAR, rv32im)\n"
                     : "ANN/SNN rv firmware v3.3 baseline (rv32im)\n");
#endif
    model_err = model_check();
    if (model_err) {
        con_puts("model: INVALID (magic/shape check failed)\n");
#if FW_MURISCV
    } else if (model->magic == ANN3_MAGIC) {   // weights stay int8: no model_prepare
        con_puts("model: ANN muRISCV-NN s8 (int8 in) ");
        con_putdec(model->n_in);
        con_puts(", ");
        con_putdec(model->n_layers);
        con_puts(" layers -> ");
        con_putdec(ann3_n_out());
        con_puts("\nready\n");
#endif
    } else {
        model_prepare();
        con_puts(model->magic == GRU_MAGIC ? "model: GRU (streaming) "
                 : model->magic == RSNN_MAGIC ? "model: RSNN (recurrent, streaming) "
                 : model_is_snn() ? "model: SNN " : "model: ANN ");
        if (model->magic == SNN_DIRECT_MAGIC || model->magic == ANN_S8_MAGIC)
            con_puts("(int8 in) ");
        con_putdec(model->n_in);
        for (u32 l = 0; l < model->n_layers; l++) {
            con_putc('-');
            con_putdec(model->layer[l].n_out);
        }
        if (model_is_snn()) {
            con_puts(" LIF, T=");
            con_putdec(model->timesteps);
        } else {
            con_puts(" ReLU");
        }
        con_puts("\nready\n");
    }

    for (;;)
        mb_service();
}
