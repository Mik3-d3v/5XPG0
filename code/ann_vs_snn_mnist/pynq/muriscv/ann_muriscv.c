// ann_muriscv.c — run an "ANN3" model (CMSIS-NN / muRISCV-NN s8 layout) with
// the UNMODIFIED muRISCV-NN fully-connected kernel, layer by layer.
//
// The same file is compiled into the board firmware (level "muriscv") and
// into the host reference (host_ref.c), so the PC reference is the same code
// as the board, built for x86 instead of RV32IM.
//
// ANN3 model.bin (little endian, written by pynq/muriscv_ref.py):
//   u32 magic "ANN3", n_layers, n_in, 0
//   per layer: u32 n_in, n_out; s32 multiplier (Q31), shift (CMSIS: > 0 left);
//              u32 w_off, b_off, w_stride (= n_in: weights are row-major
//              [n_out][n_in], one row per output, as CMSIS-NN expects)
//   weights int8 [n_out][n_in]; biases int32 [n_out], followed by
//   int32 input_offset, output_offset, act_min, act_max for that layer.
// Per layer: out = clamp(requantize(sum_j (x_j + input_offset) * w_ij + b_i)
//                        + output_offset, act_min, act_max), int8.

#include <stdint.h>

#include "muriscv_nn_functions.h"
#include "ann_muriscv.h"

typedef struct {
    uint32_t n_in, n_out;
    int32_t  multiplier, shift;
    uint32_t w_off, b_off, w_stride;
} m3_layer_t;

typedef struct {
    uint32_t magic, n_layers, n_in, timesteps;
    m3_layer_t layer[ANN3_MAX_LAYERS];
} m3_hdr_t;

int ann3_check(const uint8_t *base, uint32_t max_bytes)
{
    const m3_hdr_t *h = (const m3_hdr_t *)base;
    if (h->magic != ANN3_MAGIC || h->n_layers == 0 || h->n_layers > ANN3_MAX_LAYERS ||
        h->n_in > ANN3_MAX_WIDTH)
        return -1;
    uint32_t n_prev = h->n_in;
    for (uint32_t l = 0; l < h->n_layers; l++) {
        const m3_layer_t *d = &h->layer[l];
        if (d->n_in != n_prev || d->n_out == 0 || d->n_out > ANN3_MAX_WIDTH ||
            d->w_stride != d->n_in || ((d->w_off | d->b_off) & 3) ||
            d->w_off + d->n_in * d->n_out > max_bytes ||
            d->b_off + 4 * (d->n_out + 4) > max_bytes)
            return -1;
        n_prev = d->n_out;
    }
    return 0;
}

int32_t ann3_run(const uint8_t *base, const int8_t *input, int8_t *buf0, int8_t *buf1,
                 int32_t *logits, uint32_t *n_nonzero_hidden)
{
    const m3_hdr_t *h = (const m3_hdr_t *)base;
    const int8_t *x = input;
    int8_t *y = buf0;
    uint32_t nz = 0;
    muriscv_nn_context ctx = {0, 0};             // scalar path: no buffer needed

    for (uint32_t l = 0; l < h->n_layers; l++) {
        const m3_layer_t *d = &h->layer[l];
        const int32_t *bias = (const int32_t *)(base + d->b_off);
        const int32_t *offs = bias + d->n_out;   // in_off, out_off, act_min, act_max
        muriscv_nn_fc_params fc = {0};
        fc.input_offset = offs[0];
        fc.filter_offset = 0;
        fc.output_offset = offs[1];
        fc.activation.min = offs[2];
        fc.activation.max = offs[3];
        muriscv_nn_per_tensor_quant_params q = {d->multiplier, d->shift};
        muriscv_nn_dims in_dims = {1, 1, 1, (int32_t)d->n_in};
        muriscv_nn_dims filter_dims = {(int32_t)d->n_in, 1, 1, (int32_t)d->n_out};
        muriscv_nn_dims bias_dims = {1, 1, 1, (int32_t)d->n_out};
        muriscv_nn_dims out_dims = {1, 1, 1, (int32_t)d->n_out};

        muriscv_nn_fully_connected_s8(&ctx, &fc, &q, &in_dims, x, &filter_dims,
                                      (const int8_t *)(base + d->w_off), &bias_dims,
                                      bias, &out_dims, y);

        if (l + 1 < h->n_layers) {              // hidden: count non-zero-point outputs
            for (uint32_t o = 0; o < d->n_out; o++)   // zero point = output_offset
                nz += (int32_t)y[o] != offs[1];
            x = y;
            y = (y == buf0) ? buf1 : buf0;
        }
    }
    *n_nonzero_hidden = nz;

    const uint32_t n_out = h->layer[h->n_layers - 1].n_out;
    int32_t best = 0;
    for (uint32_t o = 0; o < n_out; o++) {
        logits[o] = y[o];
        if (y[o] > y[best])
            best = (int32_t)o;                   // first maximum, as everywhere
    }
    return best;
}
