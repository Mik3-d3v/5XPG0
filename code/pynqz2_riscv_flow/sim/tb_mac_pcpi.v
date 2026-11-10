// tb_mac_pcpi.v — drives mac_pcpi like PicoRV32 does (valid held until
// ready, dropped one cycle after) and checks every returned rd against
// vectors_mac.hex (gen_vectors_mac.py); instructions marked "not ours"
// (custom-0, unused funct7) must not raise wait / ready / wr.
`timescale 1ns/1ps
module tb;
    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;
    reg pcpi_valid = 0; reg [31:0] insn = 0, rs1 = 0, rs2 = 0;
    wire wr, wt, rdy; wire [31:0] rd;
    mac_pcpi dut(.clk(clk), .resetn(resetn), .pcpi_valid(pcpi_valid), .pcpi_insn(insn),
                 .pcpi_rs1(rs1), .pcpi_rs2(rs2), .pcpi_wr(wr), .pcpi_rd(rd),
                 .pcpi_wait(wt), .pcpi_ready(rdy));
    reg [31:0] v [0:65534];
    integer i, j, n, errors = 0, checked = 0, foreign = 0, cyc, maxcyc = 0;
    initial begin
        $readmemh("vectors_mac.hex", v);
        n = 0; while (v[5 * n] !== 32'hx && 5 * n < 65530) n = n + 1;
        repeat (3) @(posedge clk); resetn = 1; @(posedge clk);
        for (i = 0; i < n; i = i + 1) begin
            insn = v[5*i]; rs1 = v[5*i+1]; rs2 = v[5*i+2];
            pcpi_valid = 1; cyc = 0;
            if (v[5*i+3] == 2) begin        // not ours: stay silent
                foreign = foreign + 1;
                for (j = 0; j < 8; j = j + 1) begin
                    @(posedge clk); #1;
                    if (wt || rdy || wr) begin
                        errors = errors + 1;
                        if (errors < 10) $display("FAIL %0d: claimed foreign insn %h", i, insn);
                    end
                end
            end else begin
                @(posedge clk); #1;
                if (!wt && !rdy) begin $display("FAIL %0d: no wait/ready", i); errors = errors + 1; end
                while (!rdy) begin @(posedge clk); #1; cyc = cyc + 1;
                    if (cyc > 1000) begin $display("FAIL %0d: hang", i); $finish; end end
                if (cyc > maxcyc) maxcyc = cyc;
                if (v[5*i+3] == 1) begin
                    checked = checked + 1;
                    if (!wr || rd !== v[5*i+4]) begin
                        errors = errors + 1;
                        if (errors < 10) $display("FAIL %0d: insn %h rs1 %h rs2 %h: rd %h wr %b, expected %h",
                                                  i, insn, rs1, rs2, rd, wr, v[5*i+4]);
                    end
                end else if (wr) begin
                    errors = errors + 1;
                    if (errors < 10) $display("FAIL %0d: insn %h wrote rd", i, insn);
                end
            end
            @(posedge clk); pcpi_valid = 0;            // core drops valid
            @(posedge clk);
        end
        $display("RESULT: %0d instructions, %0d results checked, %0d foreign, %0d errors, max %0d cycles",
                 n, checked, foreign, errors, maxcyc);
        $finish;
    end
endmodule
