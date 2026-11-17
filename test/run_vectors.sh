#!/bin/bash
# Core-level regression without cocotb: the reciprocal proof, wide operand
# coverage, and the fault-injection campaign. Needs python3 + numpy and iverilog.
set -e
cd "$(dirname "$0")"
python3 ../model/verify_recip.py
python3 ../model/verify_reference.py 11
iverilog -o tb_vectors tb_vectors.v ../src/fp32_div_core.v
iverilog -o tb_fault   tb_fault.v   ../src/fp32_div_core.v
for mode in mantissa wide; do
    python3 ../model/gen_vectors.py $mode $mode.hex
    vvp tb_vectors +VEC=$mode.hex
done
python3 ../model/gen_vectors.py finite finite.hex
vvp tb_fault +VEC=finite.hex
