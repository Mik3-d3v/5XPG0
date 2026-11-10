// ann_muriscv.h — glue between our firmware / host reference and muRISCV-NN.
#ifndef ANN_MURISCV_H
#define ANN_MURISCV_H

#include <stdint.h>

#define ANN3_MAGIC       0x334E4E41u   // "ANN3": muRISCV-NN (CMSIS-NN) s8 layout
#define ANN3_MAX_LAYERS  4
#define ANN3_MAX_WIDTH   1024          // inputs / outputs of one layer

// 0 if the ANN3 model at base is consistent (and fits max_bytes)
int ann3_check(const uint8_t *base, uint32_t max_bytes);

// one inference: int8 input -> int32 logits (the int8 outputs of the last
// layer), returns the first argmax; buf0 / buf1: ANN3_MAX_WIDTH bytes each
int32_t ann3_run(const uint8_t *base, const int8_t *input, int8_t *buf0, int8_t *buf1,
                 int32_t *logits, uint32_t *n_nonzero_hidden);

#endif
