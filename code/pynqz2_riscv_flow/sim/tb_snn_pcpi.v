// tb_snn_pcpi.v — drives snn_pcpi like PicoRV32 does (valid held until
// ready, dropped one cycle after) and checks every returned rd against
// vectors.hex (gen_vectors.py).
`timescale 1ns/1ps
module tb;
    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;
    reg pcpi_valid = 0; reg [31:0] insn = 0, rs1 = 0, rs2 = 0;
    wire wr, wt, rdy; wire [31:0] rd;
    snn_pcpi dut(.clk(clk), .resetn(resetn), .pcpi_valid(pcpi_valid), .pcpi_insn(insn),
                 .pcpi_rs1(rs1), .pcpi_rs2(rs2), .pcpi_wr(wr), .pcpi_rd(rd),
                 .pcpi_wait(wt), .pcpi_ready(rdy));
    reg [31:0] v [0:65535];
    integer i, n, errors = 0, checked = 0, cyc, maxcyc = 0;
    initial begin
        $readmemh("vectors.hex", v);
        n = 0; while (v[4 * n] !== 32'hx && 4 * n < 65532) n = n + 1;
        repeat (3) @(posedge clk); resetn = 1; @(posedge clk);
        for (i = 0; i < n; i = i + 1) begin
            insn = v[4*i]; rs1 = v[4*i+1]; rs2 = v[4*i+2];
            pcpi_valid = 1; cyc = 0;
            @(posedge clk); #1;
            if (!wt && !rdy) begin $display("FAIL %0d: no wait/ready", i); errors = errors + 1; end
            while (!rdy) begin @(posedge clk); #1; cyc = cyc + 1;
                if (cyc > 1000) begin $display("FAIL %0d: hang", i); $finish; end end
            if (cyc > maxcyc) maxcyc = cyc;
            if (v[4*i+3] != 32'hffffffff) begin
                checked = checked + 1;
                if (!wr || rd !== v[4*i+3]) begin
                    errors = errors + 1;
                    if (errors < 10) $display("FAIL %0d: insn %h rs1 %h rs2 %h: rd %h wr %b, expected %h",
                                              i, insn, rs1, rs2, rd, wr, v[4*i+3]);
                end
            end
            @(posedge clk); pcpi_valid = 0;            // core drops valid
            @(posedge clk);
        end
        $display("RESULT: %0d instructions, %0d results checked, %0d errors, max %0d cycles",
                 n, checked, errors, maxcyc);
        $finish;
    end
endmodule
