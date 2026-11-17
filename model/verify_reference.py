"""Check fp32_div_model against two independent references.

The 2^64 input space cannot be enumerated, so this is a sampled check:

  * bulk: numpy float64 division rounded to float32 by numpy. That is correctly
    rounded (53 >= 2*24 + 2, so the double rounding is harmless). Flags are
    derived from exact arithmetic, independent of the model's internals.
  * exact: a pure-integer implementation of IEEE division (Python big ints, no
    floating point at all) on a smaller sample.

Operand classes: uniform random bits, every exponent pair, subnormal-heavy,
results near the overflow and underflow boundaries, near-equal mantissas, and
boundary mantissas.

Usage: python verify_reference.py [millions_of_bulk_pairs]     (default 50)
"""
import sys, time
import numpy as np
from fp32_div_model import fp32_div, QNAN

MIN_NORMAL = 2.0 ** -126


def f64(bits):
    with np.errstate(invalid="ignore"):                    # signalling NaN inputs
        return bits.astype(np.uint32).view(np.float32).astype(np.float64)


def ref_bulk(A, B):
    af, bf = f64(A), f64(B)
    a_nan, b_nan = np.isnan(af), np.isnan(bf)
    a_snan = a_nan & (((A >> 22) & 1) == 0)
    b_snan = b_nan & (((B >> 22) & 1) == 0)
    with np.errstate(all="ignore"):
        q64 = af / bf
        r32 = q64.astype(np.float32)
    rbits = r32.view(np.uint32).astype(np.int64)
    rf = r32.astype(np.float64)

    invalid = ((af == 0) & (bf == 0)) | (np.isinf(af) & np.isinf(bf)) | a_snan | b_snan
    nan_res = np.isnan(q64) | a_nan | b_nan
    divz = (bf == 0) & np.isfinite(af) & (af != 0)
    finite_in = np.isfinite(af) & np.isfinite(bf) & (bf != 0) & (af != 0)
    with np.errstate(all="ignore"):
        inexact = finite_in & ~((rf * bf) == af)          # rf * bf is exact in float64 (48 bits)
    overflow = finite_in & np.isinf(rf)
    m, e = np.frexp(np.where(finite_in, q64, 1.0))
    r24 = np.ldexp(np.rint(m * 16777216.0), e - 24)
    tiny = finite_in & (np.abs(r24) < MIN_NORMAL)
    underflow = tiny & inexact
    flags = (invalid * 16) | (divz * 8) | (overflow * 4) | (underflow * 2) | ((inexact | overflow) * 1)
    rbits = np.where(nan_res, QNAN, rbits)
    return rbits, flags.astype(np.int64)


def ref_exact(a, b):
    """IEEE 754 binary32 division with Python integers only. Returns (bits, flags)."""
    def dec(x):
        s, e, f = x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF
        return s, e, f
    sa, ea, fa = dec(a)
    sb, eb, fb = dec(b)
    s = sa ^ sb
    a_nan, b_nan = ea == 255 and fa != 0, eb == 255 and fb != 0
    a_inf, b_inf = ea == 255 and fa == 0, eb == 255 and fb == 0
    a_zero, b_zero = ea == 0 and fa == 0, eb == 0 and fb == 0
    snan = (a_nan and not (fa >> 22)) or (b_nan and not (fb >> 22))
    if a_nan or b_nan:
        return QNAN, 16 if snan else 0
    if (a_inf and b_inf) or (a_zero and b_zero):
        return QNAN, 16
    if a_inf or b_zero:
        return (s << 31) | 0x7F800000, 0 if a_inf else 8
    if a_zero or b_inf:
        return s << 31, 0
    # value = (na * 2^xa) / (nb * 2^xb), integers
    na, xa = (fa | (1 << 23), ea - 150) if ea else (fa, -149)
    nb, xb = (fb | (1 << 23), eb - 150) if eb else (fb, -149)
    num, den = na, nb
    x = xa - xb
    if x >= 0:
        num <<= x
    else:
        den <<= -x
    # exponent e with 2^e <= num/den < 2^(e+1)
    e = num.bit_length() - den.bit_length()
    if (num << max(-e, 0)) < (den << max(e, 0)):
        e -= 1

    def round_at(lsb):
        """Round num/den to a multiple of 2^lsb, ties to even. Returns (integer multiple, inexact)."""
        n, d = (num << -lsb, den) if lsb < 0 else (num, den << lsb)
        k, r = divmod(n, d)
        if 2 * r > d or (2 * r == d and (k & 1)):
            k += 1
        return k, r != 0

    k_unb, inexact_unb = round_at(e - 23)                 # unbounded exponent, 24 bits
    e_unb = e + 1 if k_unb == (1 << 24) else e
    tiny = e_unb < -126
    lsb = max(e - 23, -149)
    k, inexact = round_at(lsb)
    if k == 0:
        return s << 31, 2 | 1
    # k * 2^lsb, k up to 2^24
    top = k.bit_length() - 1 + lsb
    if top > 127:
        return (s << 31) | 0x7F800000, 4 | 1
    if top < -126:
        bits = k                                           # subnormal: lsb is -149
    else:
        bits = ((top + 127) << 23) | ((k << (23 - (k.bit_length() - 1))) & 0x7FFFFF)
    flags = (1 if inexact else 0) | (2 if (tiny and inexact) else 0)
    return (s << 31) | bits, flags


def samples(rng, n):
    """Yield (name, A, B) operand classes of n pairs each."""
    u = lambda: rng.integers(0, 1 << 32, n, dtype=np.int64)
    yield "uniform bits", u(), u()

    def fp(e, f=None):
        f = rng.integers(0, 1 << 23, n, dtype=np.int64) if f is None else f
        return (rng.integers(0, 2, n, dtype=np.int64) << 31) | (e << 23) | f
    ea = rng.integers(0, 256, n, dtype=np.int64)
    eb = rng.integers(0, 256, n, dtype=np.int64)
    yield "all exponent pairs", fp(ea), fp(eb)
    z = np.zeros(n, dtype=np.int64)
    yield "subnormal / anything", fp(z), u()
    yield "anything / subnormal", u(), fp(z)
    yield "subnormal / subnormal", fp(z), fp(z)
    # result exponent near the underflow boundary: ea - eb + 127 around 0
    d = rng.integers(-30, 6, n, dtype=np.int64)
    eb2 = rng.integers(1, 255, n, dtype=np.int64)
    yield "underflow boundary", fp(np.clip(eb2 - 127 + d, 0, 254)), fp(eb2)
    d = rng.integers(-4, 5, n, dtype=np.int64)
    yield "overflow boundary", fp(np.clip(eb2 + 128 + d, 0, 254)), fp(eb2)
    # near-equal mantissas and boundary mantissas
    f = rng.integers(0, 1 << 23, n, dtype=np.int64)
    yield "near-equal mantissas", fp(ea, f), fp(eb, np.clip(f + rng.integers(-3, 4, n), 0, (1 << 23) - 1))
    edge = np.array([0, 1, 2, 3, 0x400000, 0x7FFFFF, 0x7FFFFE, 0x555555, 0x2AAAAA, 0x000FFF, 0x7FF000], dtype=np.int64)
    yield "boundary mantissas", fp(ea, rng.choice(edge, n)), fp(eb)
    yield "boundary divisors", fp(ea), fp(eb, rng.choice(edge, n))
    yield "power-of-two divisors", fp(ea), fp(eb, z)


def main(millions=50):
    rng = np.random.default_rng(754)
    t0 = time.time()
    chunk = 1_000_000
    rounds = max(1, millions // 11)
    total = bad_r = bad_f = 0
    examples = []
    for i in range(rounds):
        for name, A, B in samples(rng, chunk):
            rr, rf = ref_bulk(A, B)
            mr, mf = fp32_div(A, B)
            dr, df = rr != mr, rf != mf
            bad_r += int(dr.sum()); bad_f += int(df.sum()); total += A.size
            if (dr | df).any() and len(examples) < 10:
                k = int(np.argmax(dr | df))
                examples.append((name, int(A[k]), int(B[k]), int(mr[k]), int(rr[k]), int(mf[k]), int(rf[k])))
        print(f"  bulk round {i + 1}/{rounds}  elapsed {time.time() - t0:.0f}s  bad_result={bad_r} bad_flags={bad_f}", flush=True)
    print(f"\nBulk reference: {total:,} pairs, result mismatches {bad_r}, flag mismatches {bad_f}")
    for name, a, b, mr, rr, mf, rf in examples:
        print(f"  [{name}] A=0x{a:08X} B=0x{b:08X}: model 0x{mr:08X} f={mf:05b}  ref 0x{rr:08X} f={rf:05b}")

    # exact integer reference on a smaller sample of every class
    n = 20_000
    ex_total = ex_bad = 0
    for name, A, B in samples(rng, n):
        mr, mf = fp32_div(A, B)
        for a, b, r, f in zip(A.tolist(), B.tolist(), mr.tolist(), mf.tolist()):
            er, ef = ref_exact(a, b)
            ex_total += 1
            if (er, ef) != (r, f):
                ex_bad += 1
                if ex_bad <= 10:
                    print(f"  EXACT MISMATCH [{name}] A=0x{a:08X} B=0x{b:08X}: model 0x{r:08X} f={f:05b}  ref 0x{er:08X} f={ef:05b}")
    print(f"Exact integer reference: {ex_total:,} pairs, mismatches {ex_bad}")
    print(f"Done in {time.time() - t0:.0f}s")
    return bad_r == 0 and bad_f == 0 and ex_bad == 0


if __name__ == "__main__":
    raise SystemExit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 50) else 1)
