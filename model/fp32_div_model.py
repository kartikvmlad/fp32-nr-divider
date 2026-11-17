"""Bit-accurate golden model of the FP32 (IEEE 754 binary32) Newton-Raphson divider.

Every integer width and truncation here matches the RTL, so this model doubles
as the reference for the testbenches. Works on numpy int64 arrays (vectorised)
or plain ints. All intermediate products fit in 63 bits.
"""
import numpy as np

MW = 24         # mantissa bits including the hidden one
F = 28          # fractional bits of the reciprocal x (x < 1)
TE = 30         # bits of the residual e = 1 - b*x that are scanned
P = 26          # quotient bits below the binary point before normalisation
NR_ITERS = 3
BIAS = 127
EMAX = 255
QNAN = 0x7FC00000   # canonical quiet NaN
INF = 0x7F800000
HID = 1 << (MW - 1)             # hidden bit, 2^23
FMASK = HID - 1

# 16-entry seed table, 6 fractional bits: floor(64 / top of each mantissa interval),
# so the seed never exceeds 1/mb and every later value converges from below.
SEED_LUT = np.array([60, 56, 53, 51, 48, 46, 44, 42, 40, 39, 37, 36, 35, 34, 33, 32], dtype=np.int64)

FLAG_INVALID, FLAG_DIVZERO, FLAG_OVERFLOW, FLAG_UNDERFLOW, FLAG_INEXACT = 16, 8, 4, 2, 1


def clz24(f):
    """Leading zeros of a 24-bit value (f > 0)."""
    n = np.zeros_like(f)
    found = np.zeros_like(f, dtype=bool)
    for bit in range(MW - 1, -1, -1):
        hit = ((f >> bit) & 1) == 1
        n = np.where(found | hit, n, n + 1)
        found = found | hit
    return n


def unpack(x):
    s = (x >> 31) & 1
    e = (x >> 23) & 0xFF
    f = x & FMASK
    is_zero = (e == 0) & (f == 0)
    is_sub = (e == 0) & (f != 0)
    is_inf = (e == EMAX) & (f == 0)
    is_nan = (e == EMAX) & (f != 0)
    is_snan = is_nan & ((f >> 22) == 0)
    lz = np.where(is_sub, clz24(np.where(is_sub, f, 1)), 0)
    m = np.where(is_sub, f << lz, f | HID)                  # normalised 24-bit mantissa
    E = np.where(is_sub, 1 - lz, e)                         # biased exponent, may be <= 0
    return s, E, m, is_zero, is_inf, is_nan, is_snan


def recip(mb):
    """Reciprocal of the divisor mantissa: seed, then NR_ITERS residual-form iterations."""
    x = SEED_LUT[(mb - HID) >> (MW - 5)] << (F - 6)
    for _ in range(NR_ITERS):
        pr = x * mb                                             # mul: 28 x 24, b*x < 1
        e = ~(pr >> (F + MW - 1 - TE)) & ((1 << TE) - 1)        # 1 - b*x, one's complement
        x = (x * ((1 << TE) + e)) >> TE                         # mul: 28 x 31
    return x


def nr_core(ma, mb):
    """Quotient datapath exactly as the RTL computes it.

    Returns (Q, rem_nz, ok): Q = floor(ma * 2^P / mb), the nonzero-remainder flag,
    and the built-in self-check (always True without a hardware fault).

      * residual form  x <- x * (1 + e),  e = one's complement of the bits of b*x
        (no subtractor; x always stays <= 1/mb)
      * the estimate q1 = floor(ma*x * 2^P) + 1 is therefore always Q or Q + 1
        (proven for every divisor mantissa by model/verify_recip.py)
      * the remainder r = ma*2^P - q1*mb lies in [-mb, mb). The RTL forms
        t = q1*mb + (2^24 - 1) = ma*2^P + (2^24 - 1 - r): the bits above bit 24
        must be 2*ma, bit 24 is the sign of r, and the low 24 bits give r or -r - 1.
    """
    x = recip(mb)
    sh = F + MW - 1 - P
    q1 = (ma * x + (1 << sh)) >> sh                     # mul: 28 x 24, +1 at the quotient LSB
    t = q1 * mb + (1 << MW) - 1                         # mul: 27 x 24, preloaded with 2^24 - 1
    neg = (t >> MW) & 1                                 # remainder negative -> q1 is one too big
    lo = t & ((1 << MW) - 1)
    mag = np.where(neg == 1, lo, ((1 << MW) - 1) - lo)  # -r - 1, or r
    rem_nz = np.where(neg == 1, mag != mb - 1, mag != 0)
    ok = ((t >> (MW + 1)) == 2 * ma) & (mag < mb)
    return q1 - neg, rem_nz, ok


def nr_quotient(ma, mb):
    """floor(ma * 2^P / mb) and nonzero-remainder flag."""
    q, rem_nz, _ = nr_core(ma, mb)
    return q, rem_nz


def fp32_div(a, b):
    """Returns (result_bits, flags) for FP32 a / b, RNE, full subnormals, tininess after rounding."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    a, b = np.broadcast_arrays(a, b)
    sa, Ea, ma, za, ia, na, sna = unpack(a)
    sb, Eb, mb, zb, ib, nb, snb = unpack(b)
    s = sa ^ sb

    q, rem_nz = nr_quotient(ma, mb)
    E = Ea - Eb + BIAS
    norm = (q >> P) == 0                    # quotient < 1 -> shift left
    q = np.where(norm, q << 1, q)           # now leading 1 at bit P (27-bit value)
    E = np.where(norm, E - 1, E)

    # Tininess after rounding: round to 24 bits with unbounded exponent, check E stays <= 0
    keep_u = q >> 3
    rnd_u = (q >> 2) & 1
    stk_u = ((q & 3) != 0) | rem_nz
    inc_u = rnd_u & (stk_u | (keep_u & 1))
    carry_u = (keep_u + inc_u) >> MW
    tiny = (E + carry_u) <= 0

    # Denormalise for subnormal results (shift right into exponent 0)
    sh = np.clip(1 - E, 0, 31)
    shifted_out = (q & ((1 << sh) - 1)) != 0
    qd = q >> sh
    E_res = np.where(E <= 0, 0, E)

    keep = qd >> 3
    rnd = (qd >> 2) & 1
    stk = ((qd & 3) != 0) | rem_nz | shifted_out
    inexact = (rnd == 1) | stk
    keep = keep + (rnd & (stk | (keep & 1)))

    # Rounding carry: normal 2^24 -> renormalise; subnormal 2^23 -> becomes smallest normal
    ovf_m = keep >> MW
    keep = np.where(ovf_m == 1, keep >> 1, keep)
    E_res = np.where(ovf_m == 1, E_res + 1, E_res)
    E_res = np.where((E <= 0) & ((keep >> (MW - 1)) == 1), 1, E_res)

    overflow = E_res >= EMAX
    res = (s << 31) | (E_res << 23) | (keep & FMASK)
    res = np.where(overflow, (s << 31) | INF, res)
    underflow = tiny & inexact
    flags = (overflow * (FLAG_OVERFLOW | FLAG_INEXACT)) | (underflow * FLAG_UNDERFLOW) | (inexact * FLAG_INEXACT)

    # Special cases override the arithmetic path
    invalid = (za & zb) | (ia & ib) | sna | snb
    any_nan = na | nb | invalid
    divz = zb & ~za & ~ia & ~na
    inf_res = (ia & ~ib & ~nb) | divz
    zero_res = (za & ~zb & ~nb & ~na) | (ib & ~ia & ~na)

    res = np.where(zero_res, s << 31, res)
    res = np.where(inf_res, (s << 31) | INF, res)
    res = np.where(any_nan, QNAN, res)
    special = any_nan | inf_res | zero_res
    flags = np.where(special, 0, flags)
    flags = flags | (invalid * FLAG_INVALID) | (divz * FLAG_DIVZERO)
    return res.astype(np.int64), flags.astype(np.int64)


if __name__ == "__main__":
    tests = [(0x3F800000, 0x40000000, "1/2"), (0x40400000, 0x3F800000, "3/1"), (0x3F800000, 0x40400000, "1/3"),
             (0x00000001, 0x3F800000, "min_sub/1"), (0x7F7FFFFF, 0x00000001, "max/min_sub"),
             (0x00000000, 0x00000000, "0/0"), (0x3F800000, 0x00000000, "1/0"), (0x7F800000, 0x7F800000, "inf/inf")]
    for a, b, name in tests:
        r, f = fp32_div(a, b)
        with np.errstate(all="ignore"):
            ref = (np.uint32(a).view(np.float32).astype(np.float64) /
                   np.uint32(b).view(np.float32).astype(np.float64))
        print(f"{name:12s} 0x{int(r):08X}  flags={int(f):05b}  ({np.uint32(int(r)).view(np.float32)} vs numpy {ref})")
