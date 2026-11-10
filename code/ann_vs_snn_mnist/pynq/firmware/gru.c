// gru.c — integer GRU over an int8 sample stream (magic "GRU1"): the ANN-side
// counterpart of the streaming RSNN for ECG. Bit-identical to
// int_forward_gru() in quant.py (see there for the arithmetic).
//
// Per sample t:
//   ah = Wh h + bh                 one mac_layer() pass (ann.c: the same
//                                  baseline / fast kernel as the ANN, h is
//                                  int8), 3H outputs
//   ax = Wi x_t + bi               3H multiplies (1 input)
//   r, z = SIG[...], n = TANH[...] fixed-point pre-activations, LUTs in
//                                  model.bin (offset in p1 of layer 0)
//   h  = n + z (h - n)             Q15, back to int8 (scale 1/128)
// After the last sample of each of the `bins` windows the hidden state is
// kept; the readout is linear on [those states | 3 int8 RR features].
//
// Input at INPUT_BASE: T int8 samples (zero-padded to whole words), then one
// word with the 3 int8 RR features. Logits (int32) -> RESULT_BASE.

#include "board.h"
#include "hal.h"
#include "snn_layout.h"

#ifndef FW_FAST
#define FW_FAST 0
#endif

typedef int8_t  s8;
typedef int16_t s16;
typedef int64_t s64;

#if FW_FAST
#define WVAL(x) ((s32)(x) - 128)               // offset binary (model_prepare)
#else
#define WVAL(x) ((s32)(s8)(x))
#endif

#define GRU_LUT_N 512

static s8  h[GRU_H_MAX] __attribute__((aligned(4)));   // hidden state, int8
static s32 ah[3 * GRU_H_MAX];                           // Wh h + bh
static s8  hcat[GRU_BINS_MAX * GRU_H_MAX];              // states at window ends

u32 gru_check(void)
{
    const layer_desc_t *d0 = &model->layer[0], *d1 = &model->layer[1],
                       *d2 = &model->layer[2], *d3 = &model->layer[3];
    const u32 H = d0->n_in;
    if (model->n_layers != 4 || H == 0 || H > GRU_H_MAX || (H & 3) ||
        d0->n_out != 3 * H || d1->n_in != 1 || d1->n_out != 3 * H ||
        d1->p1 == 0 || d1->p1 > GRU_BINS_MAX || d2->n_in != d1->p1 * H ||
        d3->n_in > 4 || d2->n_out != d3->n_out || d2->n_out > MAX_OUT ||
        d0->p1 < 0 || (u32)d0->p1 + 4 * GRU_LUT_N > MODEL_MAX_BYTES ||
        (d0->p1 & 3))
        return SNN_ERR_MODEL;
    for (u32 l = 0; l < 4; l++) {
        const layer_desc_t *d = &model->layer[l];
        if (d->n_out > d->w_stride || d->w_stride > MAX_NEURONS ||
            (d->w_stride & 15) || ((d->w_off | d->b_off) & 3) ||
            d->w_off + d->n_in * d->w_stride > MODEL_MAX_BYTES ||
            d->b_off + 4 * d->n_out > MODEL_MAX_BYTES)
            return SNN_ERR_MODEL;
    }
    return SNN_OK;
}

static inline u32 lut_idx(s32 p)
{
    p >>= 3;                                   // Q8 -> steps of 1/32
    if (p < -GRU_LUT_N / 2)
        p = -GRU_LUT_N / 2;
    if (p > GRU_LUT_N / 2 - 1)
        p = GRU_LUT_N / 2 - 1;
    return (u32)(p + GRU_LUT_N / 2);
}

u32 gru_run(u32 T, u32 *n_nonzero_hidden)
{
    const layer_desc_t *d0 = &model->layer[0], *d1 = &model->layer[1],
                       *d2 = &model->layer[2], *d3 = &model->layer[3];
    const u32 H = d0->n_in, bins = d1->p1;
    const s32 MH = d0->p0, MX = d1->p0;
    const s16 *SIG = (const s16 *)(MODEL_BASE + d0->p1);
    const s16 *TNH = SIG + GRU_LUT_N;
    const u8  *Wi = (const u8 *)(MODEL_BASE + d1->w_off);
    const s32 *bi = (const s32 *)(MODEL_BASE + d1->b_off);
    const volatile s8 *x = (const volatile s8 *)INPUT_BASE;
    const volatile s8 *rr = (const volatile s8 *)(INPUT_BASE + ((T + 3) & ~3u));
    u32 nz = 0, k = 0;

    for (u32 o = 0; o < H; o++)
        h[o] = 0;
    for (u32 t = 0; t < T; t++) {
        mac_layer(d0, (const u8 *)h, 1, ah);   // ah = Wh h + bh
        const s32 xt = x[t];
        for (u32 o = 0; o < H; o++) {
            const s32 axr = WVAL(Wi[o]) * xt + bi[o];
            const s32 axz = WVAL(Wi[H + o]) * xt + bi[H + o];
            const s32 axn = WVAL(Wi[2 * H + o]) * xt + bi[2 * H + o];
            const s32 pr = (s32)(((s64)ah[o] * MH + (s64)axr * MX) >> 16);
            const s32 pz = (s32)(((s64)ah[H + o] * MH + (s64)axz * MX) >> 16);
            const s32 r = SIG[lut_idx(pr)], z = SIG[lut_idx(pz)];
            const s32 pn = (s32)(((s64)axn * MX +
                                  (((s64)r * ah[2 * H + o]) >> 15) * MH) >> 16);
            const s32 n = TNH[lut_idx(pn)];
            const s32 h15 = (s32)h[o] * 256;
            const s32 hn = n + (s32)(((s64)z * (h15 - n)) >> 15);
            s32 q = (hn + 128) >> 8;
            if (q > 127)
                q = 127;
            if (q < -127)
                q = -127;
            h[o] = (s8)q;                      // ah was computed from the old h
            nz += q != 0;
        }
        if (t == T - 1 || (t + 1) * bins / T != t * bins / T) {   // window end
            for (u32 o = 0; o < H; o++)
                hcat[k * H + o] = h[o];
            k++;
        }
    }
    *n_nonzero_hidden = nz;

    // readout: (W2^T hcat) * m1 + (W3^T rr) * m2 + b
    const u8 *W2 = (const u8 *)(MODEL_BASE + d2->w_off);
    const u8 *W3 = (const u8 *)(MODEL_BASE + d3->w_off);
    const s32 *b = (const s32 *)(MODEL_BASE + d2->b_off);
    volatile s32 *logits = (volatile s32 *)RESULT_BASE;
    u32 best = 0;
    s32 best_v = 0;
    for (u32 c = 0; c < d2->n_out; c++) {
        s32 a1 = 0, a2 = 0;
        for (u32 j = 0; j < d2->n_in; j++)
            a1 += (s32)hcat[j] * WVAL(W2[j * d2->w_stride + c]);
        for (u32 j = 0; j < d3->n_in; j++)
            a2 += (s32)rr[j] * WVAL(W3[j * d3->w_stride + c]);
        s32 v = a1 * d2->p0 + a2 * d3->p0 + b[c];
        logits[c] = v;
        if (c == 0 || v > best_v) {
            best = c;
            best_v = v;
        }
    }
    return best;
}
