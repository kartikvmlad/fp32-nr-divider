"""Proof of the minimum-precision quotient datapath for FP32.

The mantissa-pair space is 2^46, too large to enumerate. The proof is instead:

  1. Exhaustive over all 2^23 divisor mantissas mb: run the reciprocal exactly as
     the RTL does and check that b*x < 1 and x < 1 at every step (so the
     residual is unsigned), and compute  D = 2^(P+s) - x*mb >= 0,  s = F + 23 - P.

  2. For a fixed mb the estimate error is linear in the dividend:
         ma*2^P/mb - ma*x/2^s  =  ma * D / (mb * 2^s)
     It is >= 0, and largest at ma = 2^24 - 1. If it is < 1 there, then for
     every ma   floor(ma*x/2^s)  is Q or Q - 1,  so  q1 = floor(ma*x/2^s) + 1
     is Q or Q + 1 - which is all the remainder correction needs.

So checking  (2^24 - 1) * D < mb * 2^s  for every mb covers every mantissa pair.
The script also cross-checks nr_core against exact integer division on a few
million random pairs plus the extreme dividends for every divisor.

Run:  python verify_recip.py       (about a minute)
"""
import numpy as np
from fp32_div_model import recip, nr_core, SEED_LUT, MW, F, TE, P, NR_ITERS, HID

CHUNK = 1 << 20


def main():
    s = F + MW - 1 - P
    ok_range = ok_bound = True
    worst = 0.0
    bad_q = bad_r = false_alarm = 0
    checked = 0
    rng = np.random.default_rng(32)
    for lo in range(HID, 2 * HID, CHUNK):
        mb = np.arange(lo, lo + CHUNK, dtype=np.int64)
        # --- step 1: reciprocal invariants, exhaustive over mb
        x = SEED_LUT[(mb - HID) >> (MW - 5)] << (F - 6)
        for _ in range(NR_ITERS):
            pr = x * mb
            ok_range &= bool((pr < (1 << (F + MW - 1))).all())
            e = ~(pr >> (F + MW - 1 - TE)) & ((1 << TE) - 1)
            x = (x * ((1 << TE) + e)) >> TE
            ok_range &= bool((x < (1 << F)).all())
        assert (x == recip(mb)).all()
        D = (1 << (P + s)) - x * mb
        # --- step 2: error bound at the largest dividend
        small = (D >= 0) & (D < (1 << 30))
        ok_bound &= bool(small.all())
        ok_bound &= bool((np.minimum(D, 1 << 30) * ((1 << MW) - 1) < (mb << s)).all())
        worst = max(worst, float((D * ((1 << MW) - 1) / (mb.astype(np.float64) * (1 << s))).max()))
        # --- cross-check against exact division: extreme and random dividends
        for ma in (np.full_like(mb, HID), np.full_like(mb, 2 * HID - 1), mb,
                   rng.integers(HID, 2 * HID, mb.size, dtype=np.int64)):
            q, rem_nz, chk = nr_core(ma, mb)
            # exact floor(ma * 2^P / mb): ma << P fits in 50 bits
            Q = (ma << P) // mb
            R = (ma << P) - Q * mb
            bad_q += int((q != Q).sum())
            bad_r += int((rem_nz != (R != 0)).sum())
            false_alarm += int((~chk).sum())
            checked += mb.size

    print(f"divisor mantissas checked    : {HID:,} (all)")
    print(f"range invariants hold        : {ok_range}")
    print(f"error bound < 1 for every mb : {ok_bound}   (worst case {worst:.4f})")
    print(f"pairs cross-checked exactly  : {checked:,}")
    print(f"quotient mismatches          : {bad_q}")
    print(f"sticky mismatches            : {bad_r}")
    print(f"self-check false alarms      : {false_alarm}")
    return ok_range and ok_bound and bad_q == 0 and bad_r == 0 and false_alarm == 0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
