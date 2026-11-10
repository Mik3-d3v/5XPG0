# firmware Makefile — MNIST ANN/SNN firmware (PicoRV32, RV32IM).
# Reuses crt0, linker script, board.h and hal.h of the bring-up flow.
# Builds three images from main.c (SNN, RSNN) + ann.c (ANN) + gru.c (GRU),
# identical results:
#   snn_mnist.bin        baseline kernels
#   snn_mnist_fast.bin   sparse/delta + SWAR kernels (-DFW_FAST=1)
#   snn_mnist_hw.bin     fast, SNN kernel on the SNN coprocessor (custom-0
#                        instructions, needs spike_top_snn.bit); the ANN
#                        code is the same as in snn_mnist_fast.bin
#   snn_mnist_muriscv.bin  fast + "ANN3" models on the UNMODIFIED muRISCV-NN
#                        library (third_party/muriscv-nn, scalar C path,
#                        external ANN reference; glue in ../muriscv/), plain
#                        rv32im, spike_top.bit
#   snn_mnist_mac.bin    hw + the ANN MAC layer on the MAC coprocessor
#                        (custom-1, ann.c FW_MAC, needs spike_top_both.bit):
#                        ANN / GRU on the MAC unit, SNN on the SNN unit (its
#                        direct-encoded first layer on the MAC unit)
# Both images run either model; the type comes from the magic of model.bin.

FLOW    ?= ../../../pynqz2_riscv_flow/firmware

CROSS   ?= riscv64-unknown-elf-
CC      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy
SIZE    := $(CROSS)size

ARCHFLAGS := -march=rv32im -mabi=ilp32
# no-tree-loop-distribute-patterns: keep gcc from turning loops into memset
CFLAGS    := $(ARCHFLAGS) -O2 -Wall -Wextra -Werror -ffreestanding -fno-builtin \
             -fno-tree-loop-distribute-patterns -nostdlib -nostartfiles -g \
             -I. -I$(FLOW)
LDFLAGS   := $(ARCHFLAGS) -T $(FLOW)/linker.ld -static -nostdlib -Wl,--build-id=none \
             -Wl,--gc-sections

HDRS := snn_layout.h $(FLOW)/board.h $(FLOW)/hal.h

# muRISCV-NN: only the files the s8 fully-connected layer needs, our -march /
# -mabi / -O flags, no -Werror (third-party code, not modified); the picolibc
# headers (picolibc-riscv64-unknown-elf) provide the library's #includes.
# USE_VEXT / USE_PEXT / USE_COREV / USE_MNN stay undefined: scalar C path.
MNN     ?= ../../../../third_party/muriscv-nn
GLUE    := ../muriscv
PICOLIBC_INC ?= /usr/lib/picolibc/riscv64-unknown-elf/include
MNN_INC := -I$(MNN)/Include -I$(GLUE) -isystem $(PICOLIBC_INC)
MNN_CFLAGS := $(filter-out -Werror,$(CFLAGS)) $(MNN_INC)
MNN_OBJS := mnn_fully_connected_s8.o mnn_fully_connected_get_buffer_sizes_s8.o \
            mnn_vec_mat_mult_t_s8.o

all: snn_mnist.bin snn_mnist_fast.bin snn_mnist_hw.bin snn_mnist_muriscv.bin      snn_mnist_mac.bin

start.o: $(FLOW)/start.S
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

%_fast.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -DFW_FAST=1 -c -o $@ $<

main_hw.o: main.c $(HDRS)
	$(CC) $(CFLAGS) -DFW_FAST=1 -DFW_HW=1 -c -o $@ $<

main_mac.o: main.c $(HDRS)
	$(CC) $(CFLAGS) -DFW_FAST=1 -DFW_HW=1 -DFW_MAC=1 -c -o $@ $<

ann_mac.o: ann.c $(HDRS)
	$(CC) $(CFLAGS) -DFW_FAST=1 -DFW_MAC=1 -c -o $@ $<

main_muriscv.o: main.c $(HDRS) $(GLUE)/ann_muriscv.h
	$(CC) $(CFLAGS) -DFW_FAST=1 -DFW_MURISCV=1 -I$(GLUE) -c -o $@ $<

ann_muriscv.o: $(GLUE)/ann_muriscv.c $(GLUE)/ann_muriscv.h
	$(CC) $(CFLAGS) $(MNN_INC) -c -o $@ $<

mnn_%.o: $(MNN)/Source/FullyConnectedFunctions/muriscv_nn_%.c
	$(CC) $(MNN_CFLAGS) -c -o $@ $<

mnn_%.o: $(MNN)/Source/NNSupportFunctions/muriscv_nn_%.c
	$(CC) $(MNN_CFLAGS) -c -o $@ $<

snn_mnist.elf: start.o main.o ann.o gru.o $(FLOW)/linker.ld
	$(CC) $(LDFLAGS) -Wl,-Map=snn_mnist.map start.o main.o ann.o gru.o -o $@ -lgcc
	$(SIZE) $@

snn_mnist_fast.elf: start.o main_fast.o ann_fast.o gru_fast.o $(FLOW)/linker.ld
	$(CC) $(LDFLAGS) -Wl,-Map=snn_mnist_fast.map start.o main_fast.o ann_fast.o gru_fast.o -o $@ -lgcc
	$(SIZE) $@

snn_mnist_hw.elf: start.o main_hw.o ann_fast.o gru_fast.o $(FLOW)/linker.ld
	$(CC) $(LDFLAGS) -Wl,-Map=snn_mnist_hw.map start.o main_hw.o ann_fast.o gru_fast.o -o $@ -lgcc
	$(SIZE) $@

snn_mnist_mac.elf: start.o main_mac.o ann_mac.o gru_fast.o $(FLOW)/linker.ld
	$(CC) $(LDFLAGS) -Wl,-Map=snn_mnist_mac.map start.o main_mac.o ann_mac.o gru_fast.o -o $@ -lgcc
	$(SIZE) $@

snn_mnist_muriscv.elf: start.o main_muriscv.o ann_fast.o gru_fast.o ann_muriscv.o $(MNN_OBJS) $(FLOW)/linker.ld
	$(CC) $(LDFLAGS) -Wl,-Map=snn_mnist_muriscv.map start.o main_muriscv.o ann_fast.o \
	    gru_fast.o ann_muriscv.o $(MNN_OBJS) -o $@ -lgcc
	$(SIZE) $@ ann_muriscv.o $(MNN_OBJS)

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f *.o *.elf *.bin *.map

.PHONY: all clean
