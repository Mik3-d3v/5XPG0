// snn_pcpi.v — SNN neuron coprocessor on the PicoRV32 PCPI interface.
//
// Custom instructions (R-type, opcode custom-0 = 0001011, op in funct7):
//
//   funct7  name        operands                        effect
//   0x00    snn.cfg     f3=slot, rs1=beta_q16, rs2=thr  beta[slot], thr[slot]
//   0x01    snn.clr     rs1=first, rs2=count            mem = cur = 0 for the range
//   0x02    snn.setcur  rs1=neuron, rs2=value           cur[neuron] = value
//   0x03    snn.acc4    rs1=neuron (multiple of 4),     cur[n+k] += w_k, k = 0..3,
//                       rs2=4 packed int8 weights       w_k = byte k of rs2
//                       f3[0]: bytes are offset binary (w + 128)
//                       f3[1]: subtract instead of add
//   0x04    snn.step    f3=slot, rs1=first,             LIF update of neurons
//                       rs2=count (<= 32) -> rd         first .. first+count-1;
//                                                       rd bit k = spike of first+k
//   0x05    snn.get     f3[0]: 0 mem / 1 cur, rs1=n     rd = mem[n] / cur[n]
//
// LIF update, bit-identical to lif() in firmware/main.c and to
// int_forward_snn() in quant.py (mem_prev > thr <=> spiked last step):
//   v = ((beta_q16 * mem) >>> 16) + cur - (mem > thr ? thr : 0)
//   mem = v;  spike = v > thr
// The currents persist across steps (delta currents): the firmware only adds
// or subtracts the weight rows of inputs that changed.
//
// snn.step is a 4-stage pipeline (read, leak multiply, integrate, compare +
// write back), one neuron per cycle: count + 4 cycles. snn.clr: count
// cycles; the others 1 cycle. pcpi_wait is raised at once, so the CPU never
// times out. Each state memory has one read address (i0) and one write port,
// so it maps to distributed (LUT) RAM.

module snn_pcpi #(
    parameter integer N = 256              // neurons (indices 0..N-1)
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

    localparam OP_CFG = 7'h00, OP_CLR = 7'h01, OP_SETCUR = 7'h02,
               OP_ACC4 = 7'h03, OP_STEP = 7'h04, OP_GET = 7'h05;
    localparam S_IDLE = 2'd0, S_RUN = 2'd1, S_DONE = 2'd2;

    // ------------------------------------------------------------------
    // state memories: membrane potentials, input currents (4 banks so that
    // acc4 updates 4 neurons per cycle), layer parameters
    // ------------------------------------------------------------------
    (* ram_style = "distributed" *) reg [31:0] mem  [0:N-1];
    (* ram_style = "distributed" *) reg [31:0] cur0 [0:N/4-1];
    (* ram_style = "distributed" *) reg [31:0] cur1 [0:N/4-1];
    (* ram_style = "distributed" *) reg [31:0] cur2 [0:N/4-1];
    (* ram_style = "distributed" *) reg [31:0] cur3 [0:N/4-1];
    reg [16:0] beta [0:3];                 // Q16, 0 .. 65536
    reg [31:0] thr  [0:3];

    integer k;
    initial begin
        for (k = 0; k < N; k = k + 1) mem[k] = 0;
        for (k = 0; k < N / 4; k = k + 1) begin
            cur0[k] = 0; cur1[k] = 0; cur2[k] = 0; cur3[k] = 0;
        end
    end

    // ------------------------------------------------------------------
    // handshake, latched operands
    // ------------------------------------------------------------------
    wire ours = pcpi_valid && pcpi_insn[6:0] == 7'b0001011 &&
                pcpi_insn[31:25] <= OP_GET;
    reg  [1:0]  st;
    reg  [6:0]  op;
    reg  [2:0]  f3;
    reg  [31:0] a, b;                      // rs1, rs2
    reg  [AW:0] cnt;                       // neurons issued / cleared so far
    assign pcpi_wait = (st == S_IDLE && ours) || st == S_RUN;

    wire [AW-1:0] i0 = a[AW-1:0] + cnt[AW-1:0];   // the one read address
    wire [AW-3:0] q0 = i0[AW-1:2];
    wire [1:0]    slot = f3[1:0];

    // asynchronous reads (distributed RAM)
    wire signed [31:0] mem_rd = mem[i0];
    wire signed [31:0] c0 = cur0[q0], c1x = cur1[q0], c2x = cur2[q0], c3x = cur3[q0];
    wire signed [31:0] cur_rd = i0[1:0] == 2'd0 ? c0 : i0[1:0] == 2'd1 ? c1x :
                                i0[1:0] == 2'd2 ? c2x : c3x;
    wire signed [31:0] thr_s = thr[slot];

    // acc4: byte k of rs2 as a signed weight
    function signed [31:0] wbyte(input [7:0] x, input offset);
        wbyte = offset ? $signed({{24{~x[7]}}, ~x[7], x[6:0]})   // x - 128
                       : $signed({{24{x[7]}}, x});
    endfunction
    wire signed [31:0] w0 = wbyte(b[7:0],   f3[0]), w1 = wbyte(b[15:8],  f3[0]),
                       w2 = wbyte(b[23:16], f3[0]), w3 = wbyte(b[31:24], f3[0]);
    function signed [31:0] addsub(input signed [31:0] x, input signed [31:0] w,
                                  input sub);
        addsub = sub ? x - w : x + w;
    endfunction

    // ------------------------------------------------------------------
    // step pipeline registers
    // ------------------------------------------------------------------
    reg              v1, v2, v3;           // stage valid
    reg  [AW-1:0]    i3, i2, i1;
    reg  [4:0]       k1, k2, k3;           // bit position in the result mask
    reg  signed [31:0] m1, cc1, m2, cc2, sum3;
    reg  signed [48:0] p2;                 // mem * beta
    reg  [31:0]      mask;
    wire issue = st == S_RUN && op == OP_STEP && cnt < b[5:0] && cnt < 32;

    // ------------------------------------------------------------------
    // memory write ports (one per memory)
    // ------------------------------------------------------------------
    reg              mem_we;
    reg  [AW-1:0]    mem_wa;
    reg  [31:0]      mem_wd;
    reg  [3:0]       cur_we;
    reg  [31:0]      cw0, cw1, cw2, cw3;
    always @* begin
        mem_we = 0; mem_wa = i3; mem_wd = sum3;
        cur_we = 0; cw0 = 0; cw1 = 0; cw2 = 0; cw3 = 0;
        if (st == S_RUN) case (op)
            OP_STEP: mem_we = v3;
            OP_CLR: if (cnt != b[AW:0]) begin
                mem_we = 1; mem_wa = i0; mem_wd = 0;
                cur_we = 4'b0001 << i0[1:0];
            end
            OP_SETCUR: begin
                cur_we = 4'b0001 << i0[1:0];
                cw0 = b; cw1 = b; cw2 = b; cw3 = b;
            end
            OP_ACC4: begin
                cur_we = 4'b1111;
                cw0 = addsub(c0, w0, f3[1]);  cw1 = addsub(c1x, w1, f3[1]);
                cw2 = addsub(c2x, w2, f3[1]); cw3 = addsub(c3x, w3, f3[1]);
            end
            default: ;
        endcase
    end
    always @(posedge clk) begin
        if (mem_we)    mem[mem_wa] <= mem_wd;
        if (cur_we[0]) cur0[q0] <= cw0;
        if (cur_we[1]) cur1[q0] <= cw1;
        if (cur_we[2]) cur2[q0] <= cw2;
        if (cur_we[3]) cur3[q0] <= cw3;
    end

    // ------------------------------------------------------------------
    // control
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        pcpi_ready <= 0;
        pcpi_wr    <= 0;
        // step pipeline (advances every cycle; only valid during snn.step)
        v1 <= issue;
        if (issue) begin
            i1 <= i0; k1 <= cnt[4:0]; m1 <= mem_rd; cc1 <= cur_rd;
        end
        v2 <= v1; i2 <= i1; k2 <= k1; m2 <= m1; cc2 <= cc1;
        p2 <= m1 * $signed({1'b0, beta[slot]});
        v3 <= v2; i3 <= i2; k3 <= k2;
        sum3 <= p2[47:16] + cc2 - (m2 > thr_s ? thr_s : 32'sd0);
        if (st == S_RUN && op == OP_STEP && v3 && $signed(sum3) > thr_s)
            mask[k3] <= 1'b1;

        if (!resetn) begin
            st <= S_IDLE;
            v1 <= 0; v2 <= 0; v3 <= 0;
        end else case (st)
        S_IDLE:
            if (ours) begin
                op  <= pcpi_insn[31:25];
                f3  <= pcpi_insn[14:12];
                a   <= pcpi_rs1;
                b   <= pcpi_rs2;
                cnt <= 0;
                mask <= 0;
                st  <= S_RUN;
            end
        S_RUN:
            case (op)
            OP_CFG: begin
                beta[slot] <= a[16:0];
                thr[slot]  <= b;
                pcpi_ready <= 1;
                st <= S_DONE;
            end
            OP_GET: begin
                pcpi_rd <= f3[0] ? cur_rd : mem_rd;
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
            OP_STEP: begin
                if (issue)
                    cnt <= cnt + 1;
                else if (!v1 && !v2 && !v3) begin   // pipeline drained
                    pcpi_rd <= mask;
                    pcpi_wr <= 1;
                    pcpi_ready <= 1;
                    st <= S_DONE;
                end
            end
            default: begin                 // setcur, acc4: written this cycle
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
