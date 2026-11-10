// tb_soc.v — whole-SoC simulation (spike_soc: PicoRV32, BRAM with its wait
// states, coprocessors per `define) running a real firmware image on
// exported samples, for checking firmware levels without the board
// (soc_sim.py writes the hex files and reads the results).
//
//   image.hex   the full 256 KB BRAM: firmware at 0, model.bin at MODEL_BASE
//   params.hex  K (samples), W (input words per sample), T (MB_CMD_RUN a0)
//   inputs.hex  K * W words
//
// As run_mnist_pynq.py on the board: the CPU boots once; per sample the
// bench writes the input to INPUT_BASE, posts MB_CMD_RUN in the mailbox,
// waits for the reply and prints it with the first 64 result words. The
// cycle count in the reply is the firmware's own timer delta, i.e. the
// number the board reports.
`timescale 1ns/1ps
module tb;
    localparam MB = 32'h10400 / 4, RES = 32'h10800 / 4, INP = 32'h11000 / 4;
    localparam MAXW = 1 << 21;
    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;
    wire [31:0] pa_rdata, timer;
    wire running, trap;
    wire [9:0] led;
    spike_soc dut(.clk(clk), .core_rst_n(rstn),
                  .pa_we(1'b0), .pa_addr(16'h0), .pa_wdata(32'h0), .pa_be(4'h0),
                  .pa_rdata(pa_rdata), .core_running(running), .core_trap(trap),
                  .timer_lo(timer), .led_o(led));
    reg [31:0] prm [0:2];
    reg [31:0] inp [0:MAXW-1];
    integer i, s, cyc, K, W;
    initial begin
        $readmemh("image.hex", dut.mem);
        $readmemh("params.hex", prm);
        $readmemh("inputs.hex", inp);
        K = prm[0]; W = prm[1];
        dut.timer_lo = 0;                  // free-running, never reset in RTL
        repeat (10) @(posedge clk);
        rstn = 1;
        for (s = 0; s < K; s = s + 1) begin
            @(posedge clk); #1;
            for (i = 0; i < W; i = i + 1)
                dut.mem[INP + i] = inp[s * W + i];
            dut.mem[MB + 2] = 16;          // MB_CMD_RUN
            dut.mem[MB + 3] = prm[2];      // a0 = T
            dut.mem[MB + 0] = s + 1;       // seq_in
            cyc = 0;
            while (dut.mem[MB + 1] !== s + 1 && !trap && cyc < 500_000_000) begin
                @(posedge clk); cyc = cyc + 1;
            end
            if (trap || cyc >= 500_000_000) begin
                $display(trap ? "TRAP" : "TIMEOUT");
                $finish;
            end
            $write("S %0d RESP %0d %0d %0d %0d OUT", s, dut.mem[MB + 6], dut.mem[MB + 7],
                   dut.mem[MB + 8], dut.mem[MB + 9]);
            for (i = 0; i < 64; i = i + 1)     // MAX_OUT; soc_sim.py keeps n_out
                $write(" %0d", $signed(dut.mem[RES + i]));
            $display("");
        end
        $display("DONE");
        $finish;
    end
endmodule
