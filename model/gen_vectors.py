"""Write test vectors for test/tb_vectors.v and test/tb_fault.v.

One 104-bit hex word per line: a (32) b (32) result (32) flags (8).

    python gen_vectors.py wide out.hex [n]     every operand class used by verify_reference.py
                                               (all exponent pairs, subnormals, overflow and
                                               underflow boundaries, boundary mantissas, ...)
    python gen_vectors.py mantissa out.hex [n] random normalised mantissa pairs plus, for every
                                               seed-table interval, its extreme divisors
    python gen_vectors.py finite out.hex [n]   random finite, non-zero operands (fault injection)
"""
import sys
import numpy as np
from fp32_div_model import fp32_div
from verify_reference import samples


def wide(n, seed=754):
    rng = np.random.default_rng(seed)
    per = max(1, n // 11)
    A, B = zip(*[(a, b) for _, a, b in samples(rng, per)])
    return np.concatenate(A), np.concatenate(B)


def mantissa(n, seed=24):
    rng = np.random.default_rng(seed)
    fa = rng.integers(0, 1 << 23, n, dtype=np.int64)
    fb = rng.integers(0, 1 << 23, n, dtype=np.int64)
    # extreme divisors of each of the 16 seed intervals, against extreme dividends
    edges = []
    for i in range(16):
        lo, hi = i << 19, ((i + 1) << 19) - 1
        edges += [lo, lo + 1, hi - 1, hi]
    eb = np.array(edges, dtype=np.int64)
    ea = np.array([0, 1, (1 << 23) - 1, (1 << 23) - 2, 0x400000, 0x2AAAAA], dtype=np.int64)
    fa = np.concatenate([fa, np.repeat(ea, eb.size)])
    fb = np.concatenate([fb, np.tile(eb, ea.size)])
    return 0x3F800000 | fa, 0x3F800000 | fb


def finite(n, seed=2026):
    rng = np.random.default_rng(seed)
    def op():
        e = rng.integers(0, 255, n, dtype=np.int64)
        f = rng.integers(0, 1 << 23, n, dtype=np.int64)
        f = np.where((e == 0) & (f == 0), 1, f)
        return (rng.integers(0, 2, n, dtype=np.int64) << 31) | (e << 23) | f
    return op(), op()


def main():
    mode, path = sys.argv[1], sys.argv[2]
    default = {"wide": 1_100_000, "mantissa": 1_000_000, "finite": 200_000}[mode]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else default
    a, b = {"wide": wide, "mantissa": mantissa, "finite": finite}[mode](n)
    res, flags = fp32_div(a, b)
    with open(path, "w") as f:
        f.write("\n".join(f"{x:08x}{y:08x}{r:08x}{g:02x}"
                          for x, y, r, g in zip(a.tolist(), b.tolist(), res.tolist(), flags.tolist())))
        f.write("\n")
    print(f"{mode}: {len(a)} vectors -> {path}")


if __name__ == "__main__":
    main()
