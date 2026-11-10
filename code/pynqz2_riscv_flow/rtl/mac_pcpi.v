// mac_pcpi.v — ANN multiply-accumulate coprocessor on the PicoRV32 PCPI
// interface: the ANN-side counterpart of snn_pcpi.v, built to the same
// pattern so that the hardware comparison has "something on both sides".
//
// Custom instructions (R-type, opcode custom-1 = 0101011, op in funct7):
//
//   funct7  name        operands                        effect
//   0x00    mac.clr     rs1=first, rs2=count            acc = 0 for the range
//   0x01    mac.setx    rs1=x                           X = x (low 9 bits, signed:
//                                                       covers int8 and uint8)
//   0x02    mac.acc4    rs1=neuron (multiple of 4),     acc[n+k] += X * w_k, k = 0..3,
//                       rs2=4 packed int8 weights       w_k = byte k of rs2
//                       f3[0]: bytes are offset binary (w + 128)
//   0x03    mac.get     rs1=neuron                      rd = acc[n]
//   0x04    mac.setacc  rs1=neuron, rs2=value           acc[n] = value (bias)
//
// mac.acc4 is the exact analogue of snn.acc4: there a spiking input adds
// (+-1 times) its weight row to the currents, here the input value X scales
// the weight row. The firmware keeps the ANN's sparse nonzero-input list,
// sets X once per nonzero input and issues one acc4 per 4 weights of its
// row, as the SNN hw kernel does per input change.
//
// Same state machine as snn_pcpi.v: mac.clr takes count cycles, the others
// 1 cycle in S_RUN; pcpi_wait is raised at once. The 4 accumulators of one
// acc4 live in 4 banks of distributed (LUT) RAM (one read address i0, one
// write port each), like cur0..cur3. One difference: the 4 products
// X * w_k (9 x 8 bit signed) are registered when the instruction is
// accepted (pcpi_rs2 and X are already stable then), so the S_RUN cycle
// only adds; this keeps multiply + 32-bit add + RAM write out of one
// 10 ns path and costs no extra cycle.

module mac_pcpi #(
    parameter integer N = 256              // accumulators (indices 0..N-1)
) (
    input             clk,
    input             resetn,
    input             pcpi_valid,
    input      [31:0] pcpi_insn,
    input      [31:0] pcpi_rs1,
    input      [31:0] pcpi_rs2,
    output reg        pcpi_wr,
    output reg [31:0] pcpi_rd,
    output            pcpi_wait,
    output reg        pcpi_ready
);
    localparam AW = $clog2(N);

    localparam OP_CLR = 7'h00, OP_SETX = 7'h01, OP_ACC4 = 7'h02,
               OP_GET = 7'h03, OP_SETACC = 7'h04;
    localparam S_IDLE = 2'd0, S_RUN = 2'd1, S_DONE = 2'd2;

    // ------------------------------------------------------------------
    // accumulators: 4 banks so that acc4 updates 4 outputs per cycle
    // ------------------------------------------------------------------
    (* ram_style = "distributed" *) reg [31:0] acc0 [0:N/4-1];
    (* ram_style = "distributed" *) reg [31:0] acc1 [0:N/4-1];
    (* ram_style = "distributed" *) reg [31:0] acc2 [0:N/4-1];
    (* ram_style = "distributed" *) reg [31:0] acc3 [0:N/4-1];
    reg signed [8:0] X;                    // current input value

    integer k;
    initial begin
        X = 0;
        for (k = 0; k < N / 4; k = k + 1) begin
            acc0[k] = 0; acc1[k] = 0; acc2[k] = 0; acc3[k] = 0;
        end
    end

    // ------------------------------------------------------------------
    // handshake, latched operands
    // ------------------------------------------------------------------
    wire ours = pcpi_valid && pcpi_insn[6:0] == 7'b0101011 &&
                pcpi_insn[31:25] <= OP_SETACC;
    reg  [1:0]  st;
    reg  [6:0]  op;
    reg  [31:0] a, b;                      // rs1, rs2
    reg  [AW:0] cnt;                       // accumulators cleared so far
    assign pcpi_wait = (st == S_IDLE && ours) || st == S_RUN;

    wire [AW-1:0] i0 = a[AW-1:0] + cnt[AW-1:0];   // the one read address
    wire [AW-3:0] q0 = i0[AW-1:2];

    // asynchronous reads (distributed RAM)
    wire signed [31:0] r0 = acc0[q0], r1 = acc1[q0], r2 = acc2[q0], r3 = acc3[q0];
    wire signed [31:0] acc_rd = i0[1:0] == 2'd0 ? r0 : i0[1:0] == 2'd1 ? r1 :
                                i0[1:0] == 2'd2 ? r2 : r3;

    // acc4: byte k of rs2 as a signed weight (as in snn_pcpi.v)
    function signed [7:0] wbyte(input [7:0] x, input offset);
        wbyte = offset ? {~x[7], x[6:0]}   // x - 128
                       : x;
    endfunction
    // products, registered at accept time (see top of file)
    reg signed [16:0] p0, p1, p2, p3;
    wire off_in = pcpi_insn[12];           // f3[0] of the incoming instruction

    // ------------------------------------------------------------------
    // accumulator write ports (one per bank)
    // ------------------------------------------------------------------
    reg  [3:0]  we;
    reg  [31:0] w0d, w1d, w2d, w3d;
    always @* begin
        we = 0; w0d = 0; w1d = 0; w2d = 0; w3d = 0;
        if (st == S_RUN) case (op)
            OP_CLR: if (cnt != b[AW:0])
                we = 4'b0001 << i0[1:0];
            OP_SETACC: begin
                we = 4'b0001 << i0[1:0];
                w0d = b; w1d = b; w2d = b; w3d = b;
            end
            OP_ACC4: begin
                we = 4'b1111;
                w0d = r0 + p0;  w1d = r1 + p1;
                w2d = r2 + p2;  w3d = r3 + p3;
            end
            default: ;
        endcase
    end
    always @(posedge clk) begin
        if (we[0]) acc0[q0] <= w0d;
        if (we[1]) acc1[q0] <= w1d;
        if (we[2]) acc2[q0] <= w2d;
        if (we[3]) acc3[q0] <= w3d;
    end

    // ------------------------------------------------------------------
    // control
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        pcpi_ready <= 0;
        pcpi_wr    <= 0;

        if (!resetn) begin
            st <= S_IDLE;
        end else case (st)
        S_IDLE:
            if (ours) begin
                op  <= pcpi_insn[31:25];
                a   <= pcpi_rs1;
                b   <= pcpi_rs2;
                cnt <= 0;
                p0  <= X * wbyte(pcpi_rs2[7:0],   off_in);
                p1  <= X * wbyte(pcpi_rs2[15:8],  off_in);
                p2  <= X * wbyte(pcpi_rs2[23:16], off_in);
                p3  <= X * wbyte(pcpi_rs2[31:24], off_in);
                st  <= S_RUN;
            end
        S_RUN:
            case (op)
            OP_SETX: begin
                X <= a[8:0];
                pcpi_ready <= 1;
                st <= S_DONE;
            end
            OP_GET: begin
                pcpi_rd <= acc_rd;
                pcpi_wr <= 1;
                pcpi_ready <= 1;
                st <= S_DONE;
            end
            OP_CLR:
                if (cnt == b[AW:0]) begin
                    pcpi_ready <= 1;
                    st <= S_DONE;
                end else
                    cnt <= cnt + 1;
            default: begin                 // setacc, acc4: written this cycle
                pcpi_ready <= 1;
                st <= S_DONE;
            end
            endcase
        S_DONE:                            // wait until the CPU drops valid
            if (!pcpi_valid)
                st <= S_IDLE;
        default:
            st <= S_IDLE;
        endcase
    end
endmodule
