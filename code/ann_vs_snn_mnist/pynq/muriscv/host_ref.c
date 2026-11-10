// host_ref.c — PC reference for the "muriscv" level: the same muRISCV-NN
// sources and the same glue (ann_muriscv.c) as the board firmware, built for
// the host. Reads model.bin (ANN3) and inputs.bin ([N][n_in] int8), writes
// host_outputs.bin ([N][n_out] int32 logits) and host_preds.bin ([N] uint8).
// Input rows are n_in bytes zero-padded to a multiple of 4, as on the board.
//
// Usage: host_ref <export_dir>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ann_muriscv.h"

static uint8_t *slurp(const char *path, long *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(*size + 16);
    if (fread(buf, 1, *size, f) != (size_t)*size) { perror(path); exit(1); }
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: %s <export_dir>\n", argv[0]); return 2; }
    char path[4096];
    long msize, isize;
    snprintf(path, sizeof path, "%s/model.bin", argv[1]);
    uint8_t *model = slurp(path, &msize);
    if (ann3_check(model, (uint32_t)msize)) { fprintf(stderr, "bad ANN3 model\n"); return 1; }
    const uint32_t *hdr = (const uint32_t *)model;
    const uint32_t n_layers = hdr[1], n_in = hdr[2];
    const uint32_t n_out = hdr[4 + 7 * (n_layers - 1) + 1];

    snprintf(path, sizeof path, "%s/inputs.bin", argv[1]);
    int8_t *inputs = (int8_t *)slurp(path, &isize);
    const uint32_t stride = (n_in + 3) & ~3u;   // rows zero-padded to words (quant.pad4)
    const long n = isize / stride;

    snprintf(path, sizeof path, "%s/host_outputs.bin", argv[1]);
    FILE *fo = fopen(path, "wb");
    snprintf(path, sizeof path, "%s/host_preds.bin", argv[1]);
    FILE *fp = fopen(path, "wb");
    static int8_t b0[ANN3_MAX_WIDTH], b1[ANN3_MAX_WIDTH];
    int32_t logits[64];
    for (long i = 0; i < n; i++) {
        uint32_t nz;
        int32_t pred = ann3_run(model, inputs + i * stride, b0, b1, logits, &nz);
        uint8_t p = (uint8_t)pred;
        fwrite(logits, sizeof(int32_t), n_out, fo);
        fwrite(&p, 1, 1, fp);
    }
    fclose(fo);
    fclose(fp);
    printf("%ld samples, %u layers, %u inputs, %u outputs\n", n, n_layers, n_in, n_out);
    return 0;
}
