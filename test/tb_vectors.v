// Drives fp32_div_core with vectors from model/gen_vectors.py and compares result,
// flags and the self-check bit (which must stay 0). Plain Verilog, no cocotb:
//
//   python ../model/gen_vectors.py wide wide.hex
//   iverilog -o tb_vectors tb_vectors.v ../src/fp32_div_core.v
//   vvp tb_vectors +VEC=wide.hex
`default_nettype none
`timescale 1ns / 1ps
module tb_vectors;
    reg         clk = 0, rst_n = 0, start = 0;
    reg  [31:0] a = 0, b = 0;
    wire [31:0] result;
    wire [4:0]  flags;
    wire        fault, busy, done;
    fp32_div_core dut (.clk(clk), .rst_n(rst_n), .start(start), .a(a), .b(b),
                       .result(result), .flags(flags), .fault(fault), .busy(busy), .done(done));
    always #5 clk = ~clk;

    reg [103:0]     vec;
    reg [1023:0]    path;
    integer fd, code, n, errors, cycles, max_cycles;
    initial begin
        if (!$value$plusargs("VEC=%s", path)) begin
            $display("usage: vvp tb_vectors +VEC=file.hex");
            $finish;
        end
        fd = $fopen(path, "r");
        if (fd == 0) begin
            $display("cannot open vector file");
            $finish;
        end
        n = 0; errors = 0; max_cycles = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        code = $fscanf(fd, "%h", vec);
        while (code == 1) begin
            @(negedge clk);
            a = vec[103:72];
            b = vec[71:40];
            start = 1;
            @(negedge clk);
            start = 0;
            cycles = 1;
            while (!done) begin @(negedge clk); cycles = cycles + 1; end
            if (cycles > max_cycles) max_cycles = cycles;
            if (result !== vec[39:8] || flags !== vec[4:0] || fault !== 1'b0) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH a=%h b=%h got %h/%b fault=%b exp %h/%b",
                             a, b, result, flags, fault, vec[39:8], vec[4:0]);
            end
            n = n + 1;
            code = $fscanf(fd, "%h", vec);
        end
        $display("vectors=%0d errors=%0d max_cycles=%0d", n, errors, max_cycles);
        $finish;
    end
endmodule
