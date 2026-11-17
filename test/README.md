# Sample testbench for a Tiny Tapeout project

This is a sample testbench for a Tiny Tapeout project. It uses [cocotb](https://docs.cocotb.org/en/stable/) to drive the DUT and check the outputs.
See below to get started or for more information, check the [website](https://tinytapeout.com/hdl/testing/).

## Setting up

1. Edit [Makefile](Makefile) and modify `PROJECT_SOURCES` to point to your Verilog files.
2. Edit [tb.v](tb.v) and replace `tt_um_example` with your module name.

## How to run

To run the RTL simulation:

```sh
make -B
```

To run gatelevel simulation, first harden your project and copy `../runs/wokwi/results/final/verilog/gl/{your_module_name}.v` to `gate_level_netlist.v`.

Then run:

```sh
make -B GATES=yes
```

If you wish to save the waveform in VCD format instead of FST format, edit tb.v to use `$dumpfile("tb.vcd");` and then run:

```sh
make -B FST=
```

This will generate `tb.vcd` instead of `tb.fst`.

## Core-level regression (no cocotb)

`run_vectors.sh` checks the divider core directly with plain Verilog testbenches. It needs only
`python3` with `numpy` and `iverilog`:

```sh
bash run_vectors.sh
```

It runs, in order:

1. `../model/verify_recip.py` - proves the minimum-precision quotient datapath in the Python model for
   every one of the 8,388,608 divisor mantissas. Re-run this after changing the seed table or any
   datapath width.
2. `../model/verify_reference.py` - compares the model with two independent references on sampled
   operands (pass a number of millions as an argument for a longer run).
3. `tb_vectors.v` on about one million random mantissa pairs plus the extreme divisors of every seed
   interval, then on about 1.1 million vectors covering every operand class (all exponent pairs,
   subnormals, overflow and underflow boundaries, infinities and NaNs). Result, flags and the self-check
   bit are compared.
4. `tb_fault.v` - flips one random working-register bit at a random clock of 200,000 divides and reports
   how many upsets were masked, detected by the self-check, or silent.

## Named internal values

RTL simulation is built with `-DFP32_DEBUG`, which adds `dbg_x0` .. `dbg_x3` and `dbg_q1` to the core.
These are named copies of the reciprocal estimates and the quotient estimate, which otherwise exist only
as bit fields of `{H, L}` at a phase boundary. `divider.gtkw` shows them.

## How to view the waveform file

Using GTKWave (`divider.gtkw` shows the core's internals, `tb.gtkw` just the pins)

```sh
gtkwave tb.fst divider.gtkw
```

Using Surfer

```sh
surfer tb.fst
```
