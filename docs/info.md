## How it works

This project is an IEEE 754 **single-precision (FP32)** floating-point divider that computes **Q = A / B**
using a **Newton-Raphson reciprocal**.

- **Format:** binary32 (1 sign, 8 exponent, 23 fraction bits)
- **Rounding:** round-to-nearest, ties-to-even (correctly rounded)
- **Subnormals:** fully supported on inputs and outputs
- **Exception flags:** invalid, divide-by-zero, overflow, underflow, inexact
- **NaN results:** canonical quiet NaN `0x7FC00000`; underflow uses tininess-after-rounding

**Algorithm**

1. Unpack A and B; normalise subnormal mantissas; detect zero, infinity and NaN.
2. Seed the reciprocal of B's mantissa from a 16-entry, 6-bit lookup table.
3. Three Newton-Raphson iterations in residual form, `x = x * (1 + e)` with `e = 1 - B * x`. The residual is
   formed by inverting the product bits, so no subtractor is needed, and `x` always approaches the
   reciprocal from below.
4. Quotient estimate `q1 = floor(A * x) + 1`. Because `x` never overshoots, `q1` is either the exact
   quotient or one more than it.
5. Remainder check from the low bits of `q1 * B` only: the true remainder lies in `[-B, B)`, so its sign and
   zero-ness are read straight from the product, and the quotient is corrected by a conditional decrement.
6. Normalise, denormalise for subnormal results, round to nearest even, pack, and set the flags.

**Datapath**

The whole computation runs on three working registers: a multiplicand `M`, an accumulator `H` and a shift
register `L` that holds the scanned operand and the low product bits. The reciprocal, residual, quotient and
remainder are never stored separately. The multiplier is radix-4 (2 bits per clock) with digit 3 recoded as
4 - 1, so it needs no 3x multiple. The operand widths are the smallest for which the result is still exact;
`model/verify_recip.py` proves this for every divisor mantissa.

A divide takes about **124 clock cycles** (up to 15 more when the result is subnormal). A power-of-two divisor
skips the reciprocal and finishes much sooner, and special cases such as 0, infinity or NaN take 1 cycle. At
33.33 MHz one divide takes roughly 4 us.

**Built-in self-check**

The remainder multiply doubles as a check on everything before it. It is biased so that, for a correct
estimate `q1`, the upper product bits equal `2 * A` exactly and the low bits lie within `+/- B`. A disturbed
reciprocal or quotient estimate that would give a wrong result sets the `fault` bit (bit 7 of the flags byte).
It is always 0 on working silicon. The check does not cover the unpacking, rounding or interface logic.

**Verification**

- `model/verify_recip.py` proves the reciprocal error bound for every divisor mantissa.
- `model/verify_reference.py` cross-checks the Python golden model against an independent reference.
- `test/run_vectors.sh` compares the RTL core against the model on mantissa, wide-exponent and special-value
  vectors, and runs a fault-injection campaign on the self-check.
- The cocotb tests in `test/test.py` drive the full chip interface: corner cases, random operands,
  subnormals, and writes or starts while a divide is running.

## How to test

The chip uses a byte-wide register interface. The 3-bit register address is `{uio_in[4], uio_in[1:0]}`.

| Pin | Function |
|---|---|
| `ui_in[7:0]` | data byte to write |
| `uio_in[1:0]`, `uio_in[4]` | register address (see below) |
| `uio_in[2]` | write strobe: one byte is written per strobe (hold address and data stable while it is high) |
| `uio_in[3]` | start (rising edge) |
| `uo_out[7:0]` | byte selected by the address |
| `uio_out[6]` | busy |
| `uio_out[7]` | done (stays high until the next start) |

| Address | Write | Read |
|---|---|---|
| 0 to 3 | A[7:0], A[15:8], A[23:16], A[31:24] | Q[7:0], Q[15:8], Q[23:16], Q[31:24] |
| 4 | B[7:0] | flags |
| 5 to 7 | B[15:8], B[23:16], B[31:24] | status |

Flags byte: `{fault, 0, 0, invalid, divzero, overflow, underflow, inexact}`. `fault` is the self-check result
and reads 0 in normal operation. Status byte: `{0, 0, 0, 0, 0, 0, busy, done}`.

**To perform one division:**

1. Reset the design (`rst_n` low, then high) and provide a clock (up to 33 MHz).
2. For each of the eight operand bytes (A low to high, then B low to high): put the byte on `ui_in`, set the
   address, raise `uio_in[2]` for at least 3 clock cycles, then lower it.
3. Pulse `uio_in[3]` (start) high for at least 3 clock cycles.
4. Wait until `uio_out[7]` (done) is high. Operands cannot change during a divide: a write strobe raised while
   `uio_out[6]` (busy) is high takes effect when the divide finishes, if it is still high then.
5. Set the address to 0, 1, 2 and 3 and read Q low to high on `uo_out`, then address 4 for the flags.

**Example values**

| A | B | Q | Flags | Meaning |
|---|---|---|---|---|
| `3F800000` (1.0) | `40000000` (2.0) | `3F000000` (0.5) | `00000` | exact |
| `3F800000` (1.0) | `40400000` (3.0) | `3EAAAAAB` (0.33333334) | `00001` | inexact |
| `3F800000` (1.0) | `00000000` (0.0) | `7F800000` (+inf) | `01000` | divide-by-zero |
| `00000000` (0.0) | `00000000` (0.0) | `7FC00000` (NaN) | `10000` | invalid |
| `7F7FFFFF` (max finite) | `00000001` (min subnormal) | `7F800000` (+inf) | `00101` | overflow + inexact |

## External hardware

None. The Tiny Tapeout demo board's microcontroller can drive the inputs and read the outputs directly.
