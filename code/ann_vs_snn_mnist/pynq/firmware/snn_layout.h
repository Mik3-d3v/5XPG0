// snn_layout.h — BRAM layout, model header and mailbox commands of the MNIST
// ANN/SNN firmware.
//
// MIRRORED in host/run_mnist_pynq.py and export_model.py — keep in sync!
// Console (0x10000) and mailbox (0x10400) are the ones of the bring-up flow
// (../../../pynqz2_riscv_flow/firmware/board.h); the spike-log ring of the
// demo is not used, its space holds the input spikes instead.
//
//   0x00000 .. 0x0FFFF  code + data + bss          (flow linker.ld)
//   0x10000             console ring               (board.h)
//   0x10400             mailbox                    (board.h)
//   0x10800             result: SNN u32 out_counts[n_out] / ANN s32 logits[n_out]
//   0x11000 .. 0x12FFF  input of one sample (PS writes):
//                         SNN2 spikes [T][words] u32 bitmaps, ANN1 u8 [n_in],
//                         ANN2 / SNN3 s8 [n_in] (zero-padded to whole words),
//                         RSN1 [T] u32 event words + 1 word of 3 int8 RR feats,
//                         GRU1 [T] int8 samples (padded) + 1 word of RR feats
//   0x13000 .. 0x3EFFF  model.bin (PS writes, see export_model.py)
//   0x3F000 .. 0x3FFFF  stack (grows down from 0x40000)

#ifndef SNN_LAYOUT_H
#define SNN_LAYOUT_H

#define RESULT_BASE      0x10800u
#define INPUT_BASE       0x11000u
#define INPUT_MAX_BYTES  0x2000u
#define MODEL_BASE       0x13000u
#define MODEL_MAX_BYTES  (0x3F000u - MODEL_BASE)

#define SNN_MAGIC        0x324E4E53u   // "SNN2" spike-bitmap inputs
#define SNN_DIRECT_MAGIC 0x334E4E53u   // "SNN3" int8 inputs, direct encoding:
                                       //   constant first-layer current
#define ANN_MAGIC        0x314E4E41u   // "ANN1" uint8 inputs
#define ANN_S8_MAGIC     0x324E4E41u   // "ANN2" int8 inputs
#define GRU_MAGIC        0x31555247u   // "GRU1" GRU over an int8 sample
                                       //   stream + linear readout (ECG)
#define GRU_H_MAX        128           // GRU hidden size
#define GRU_BINS_MAX     4             // readout windows
#define RSNN_MAGIC       0x314E5352u   // "RSN1" recurrent SNN over an event
                                       //   stream + linear readout (ECG)
#define ANN_REQ_SHIFT    24            // hidden requantization (acc * m_q) >> 24
#define MAX_LAYERS       4
#define MAX_NEURONS      1024          // per layer (incl. the 784 inputs)
#define MAX_OUT          64
#define SNN_COP_NEURONS  256           // neurons in the SNN coprocessor (snn_pcpi.v)
#define RSNN_MAX_H       256           // RSNN hidden neurons (= coprocessor size)
#define SNN_MAX_OUT      256           // outputs (w_stride) of one SNN layer

// mailbox commands (numbers above the demo's MB_CMD_* in board.h)
#define MB_CMD_RUN       16   // a0=T (SNN) -> resp0=pred resp1=cycles
                              //   resp2=SNN spikes / ANN nonzero hidden, resp3=err

// errors (resp3 of MB_CMD_SNN_RUN, also printed on the console at boot)
#define SNN_OK           0
#define SNN_ERR_MODEL    1
#define SNN_ERR_INPUT    2

// ------------------------------------------------------------------
// model.bin header (export_model.py: pack_model)
// ------------------------------------------------------------------
typedef struct {
    uint32_t n_in, n_out;
    int32_t  p0, p1;                   // SNN: beta_q16, thr   ANN: m_q, 0
    uint32_t w_off, b_off;             // bytes from MODEL_BASE
    uint32_t w_stride;                 // bytes per weight row (multiple of 16)
} layer_desc_t;

typedef struct {
    uint32_t magic, n_layers, n_in, timesteps;
    layer_desc_t layer[MAX_LAYERS];
} model_hdr_t;

static const model_hdr_t *const model = (const model_hdr_t *)MODEL_BASE;

// ann.c
uint32_t ann_run(uint32_t *n_nonzero_hidden);
// gru.c
uint32_t gru_check(void);
uint32_t gru_run(uint32_t T, uint32_t *n_nonzero_hidden);
// out = bias + x * W of one layer; x: uint8, or int8 if x_signed; x zero-
// padded to whole words. Weights as stored (fast build: offset binary).
void mac_layer(const layer_desc_t *d, const uint8_t *x, int x_signed,
               int32_t *out);

#endif // SNN_LAYOUT_H
