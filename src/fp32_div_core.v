/*
 * FP32 (IEEE 754 binary32) divider, Newton-Raphson reciprocal.
 * Round-to-nearest-even, full subnormal support, all five IEEE flags.
 * Bit-exact with model/fp32_div_model.py.
 *
 * Register-folded, minimum-precision datapath (237 cycles per divide, up to 31
 * more when the result is subnormal; special cases 1 cycle):
 *
 *   - Three working registers only: M (multiplicand), H (accumulator), L (scanned
 *     operand / low product bits). Every Newton-Raphson variable - reciprocal,
 *     residual, quotient, remainder - lives in the multiplier's own registers.
 *   - Residual-form iteration x <- x * (1 + e) with e = ~(b*x): the "2 - b*x"
 *     subtractor is a row of inverters, and x converges from below. Three
 *     iterations from a 16-entry, 6-bit seed.
 *   - One-sided estimate: q1 = floor(a*x) + 1 is always Q or Q + 1, so the
 *     correction is a single conditional decrement.
 *   - Low-bits-only remainder: a*2^26 - q1*b lies in [-b, b), so only the low
 *     25 bits of q1*b are examined; a*2^26 contributes nothing to them. No
 *     remainder register or subtractor.
 *   - Radix-2 shift-and-add multiplier: one multiplier bit per clock, so the
 *     partial product is just M or 0 - no 2*M select, no digit recoding.
 *   - One shared input normaliser: only one operand mantissa is ever needed in
 *     a given cycle, so a and b take turns on a single barrel shifter.
 *   - No output shifter: the quotient is normalised / denormalised by running
 *     it through {H, L} with zero multiplier digits, one bit per clock; the
 *     bits that fall into L are the sticky bits.
 *   - One adder: the quotient correction (-1) and the rounding increment both
 *     go through the multiplier's adder, so there is no separate decrementer
 *     or rounding incrementer.
 *   - No result register: the packed result is written back into L, which is
 *     otherwise idle between divides. "result" is therefore only meaningful
 *     from done until the next start.
 *   - Early-out: a power-of-two divisor (normal, mantissa 1.0) needs no
 *     reciprocal; the quotient is the dividend mantissa (3 cycles).
 *   - Built-in self-check: the remainder multiply is biased so that the upper
 *     product bits must equal 2*ma and the low bits must lie within +/- mb. That
 *     holds only if q1 is Q or Q + 1, so any upset in the reciprocal or quotient
 *     multiplies that would give a wrong quotient raises "fault".
 *
 * Widths (28-bit reciprocal, 30-bit residual, 16-entry 6-bit seed) are the
 * minimum for which this is exact; proven for every divisor mantissa by
 * model/verify_recip.py. Re-run it after touching the seed table or any width.
 *
 * a and b must be held stable while busy is high (the wrapper guarantees it).
 *
 * Define FP32_DEBUG in simulation to get named copies of x0..x3 and q1.
 */
`default_nettype none

module fp32_div_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result,     // valid from done until the next start
    output reg  [4:0]  flags,      // {invalid, divzero, overflow, underflow, inexact}
    output reg         fault,      // self-check failed on the last divide
    output wire        busy,
    output reg         done
);

    // Phases. M is always the multiplicand.
    localparam [3:0] PH_BX0 = 4'd0,   // x0 * mb           -> residual e
                     PH_XT0 = 4'd1,   // x0 * (1 + e)      -> x1
                     PH_BX1 = 4'd2,   // x1 * mb           -> residual e
                     PH_XT1 = 4'd3,   // x1 * (1 + e)      -> x2
                     PH_BX2 = 4'd4,   // x2 * mb           -> residual e
                     PH_XT2 = 4'd5,   // x2 * (1 + e)      -> x3
                     PH_QUO = 4'd6,   // x3 * ma (+1)      -> q1
                     PH_REM = 4'd7,   // q1 * mb (+2^24-1) -> remainder sign, range check
                     PH_CHK = 4'd8,   // one cycle: upper product bits == 2*ma, load the quotient
                     PH_DEN = 4'd9;   // shift the quotient into place, then round

    reg       run;
    reg [3:0] ph;
    reg [5:0] cnt;                 // multiply steps left in this phase; 0 = phase boundary
    assign busy = run;

    // ---------------- unpack (combinational on inputs) ----------------
    wire        sa = a[31], sb = b[31];
    wire [7:0]  ea = a[30:23], eb = b[30:23];
    wire [22:0] fa = a[22:0],  fb = b[22:0];

    wire a_fz   = (fa == 23'd0);
    wire b_fz   = (fb == 23'd0);
    wire a_zero = (ea == 8'd0)   && a_fz;
    wire b_zero = (eb == 8'd0)   && b_fz;
    wire a_sub  = (ea == 8'd0)   && !a_fz;
    wire b_sub  = (eb == 8'd0)   && !b_fz;
    wire a_inf  = (ea == 8'd255) && a_fz;
    wire b_inf  = (eb == 8'd255) && b_fz;
    wire a_nan  = (ea == 8'd255) && !a_fz;
    wire b_nan  = (eb == 8'd255) && !b_fz;
    wire a_snan = a_nan && !fa[22];
    wire b_snan = b_nan && !fb[22];

    function [4:0] clz23;          // leading zeros of a 23-bit value, as a 24-bit field (>= 1)
        input [22:0] f;
        integer i;
        begin
            clz23 = 5'd23;
            for (i = 0; i < 23; i = i + 1)
                if (f[i]) clz23 = 5'd23 - i[4:0];
        end
    endfunction

    wire [4:0]  lza = clz23(fa);
    wire [4:0]  lzb = clz23(fb);
    // biased exponents; every exponent in this module lies in -150 .. 404, so 10 bits
    wire signed [9:0] Ea = a_sub ? (10'sd1 - $signed({5'd0, lza})) : $signed({2'd0, ea});
    wire signed [9:0] Eb = b_sub ? (10'sd1 - $signed({5'd0, lzb})) : $signed({2'd0, eb});
    wire        [9:0] e_t = Ea + ~Eb;                             // Ea - Eb - 1
    wire signed [9:0] E0  = {e_t[9:7] + 3'd1, e_t[6:0]};          // + 128

    wire invalid_n = (a_zero && b_zero) || (a_inf && b_inf) || a_snan || b_snan;
    wire nan_n     = a_nan || b_nan || invalid_n;
    wire divz_n    = b_zero && !a_zero && !a_inf && !a_nan;
    wire inf_n     = (a_inf && !b_inf && !b_nan) || divz_n;
    wire zero_n    = (a_zero && !b_zero && !b_nan && !a_nan) || (b_inf && !a_inf && !a_nan);
    wire special_n = nan_n || inf_n || zero_n;
    wire s         = sa ^ sb;
    wire pow2      = b_fz && (eb != 8'd0);               // normal divisor with mantissa exactly 1.0

    // Shared normaliser. The dividend mantissa is needed when the quotient multiply
    // is loaded, in the check cycle, and at the start of an early-out; the divisor
    // mantissa everywhere else.
    wire        use_a = run ? (ph == PH_XT2 || ph == PH_CHK) : pow2;
    wire [22:0] f_s   = use_a ? fa    : fb;
    wire [4:0]  lz_s  = use_a ? lza   : lzb;
    wire        sub_s = use_a ? a_sub : b_sub;
    wire [23:0] m_s   = sub_s ? ({1'b0, f_s} << lz_s) : {1'b1, f_s};

    // seed table: floor(64 / top of mantissa interval), never above 1/mb
    function [5:0] seed;
        input [3:0] i;
        begin
            case (i)
                4'd0:  seed = 6'd60; 4'd1:  seed = 6'd56; 4'd2:  seed = 6'd53; 4'd3:  seed = 6'd51;
                4'd4:  seed = 6'd48; 4'd5:  seed = 6'd46; 4'd6:  seed = 6'd44; 4'd7:  seed = 6'd42;
                4'd8:  seed = 6'd40; 4'd9:  seed = 6'd39; 4'd10: seed = 6'd37; 4'd11: seed = 6'd36;
                4'd12: seed = 6'd35; 4'd13: seed = 6'd34; 4'd14: seed = 6'd33; default: seed = 6'd32;
            endcase
        end
    endfunction

    // ---------------- datapath: M, H, L ----------------
    // {H, L} shifts right one bit per step. 26-step phases (26-bit scanned
    // operand) leave the product in {H, L[31:6]}; 32-step phases in {H, L}.
    reg        [27:0] M;      // x (28 bits) or 2*q1
    reg signed [28:0] H;
    reg        [31:0] L;
    reg               r_neg;  // remainder < 0: q1 = Q + 1
    reg               rem_nz; // remainder of the corrected quotient is non-zero
    reg               nrm_r;  // quotient < 1: result exponent is one lower
    reg               tiny_r; // result is tiny after rounding (underflow if also inexact)

    // The one adder. Normally a radix-2 multiply step: H + (L[0] ? M : 0). Two other uses:
    //   chk : H input replaced by -2*r_neg, digit forced to 1  ->  M - 2*r_neg = 2*Q
    //   fin : digit is 0, rounding increment injected at bit 3 ->  H + 8*round
    wire        chk   = run && (cnt == 6'd0) && (ph == PH_CHK);
    wire        fin   = run && (cnt == 6'd0) && (ph == PH_DEN);
    wire        rcm   = run && (cnt == 6'd0) && (ph == PH_REM);   // remainder comparison
    wire        d_one = (L[0] & ~fin) | chk;
    wire        rnd_inc;
    wire [23:0] r_mag;
    wire [30:0] pp    = rcm ? {7'd0, ~r_mag} : ((d_one ? {3'b000, M} : 31'd0) | {27'd0, rnd_inc, 3'd0});
    wire [28:0] h_in  = chk ? {{28{r_neg}}, 1'b0} : rcm ? {5'd0, m_s} : H;
    wire [30:0] sum   = {{2{h_in[28]}}, h_in} + pp;

    // values handed from one phase to the next
    wire [29:0] resid = ~{H[24:0], L[31:27]};            // 1 - b*x, bits 2^-1 .. 2^-30
    wire        to_xt = (ph == PH_BX0) | (ph == PH_BX1) | (ph == PH_BX2);   // next phase scans 1 + e
    wire        q_now = (ph == PH_QUO);                  // this phase produced q1
    wire        x_end = (ph == PH_XT2);                  // next phase is the quotient multiply

    // ---------------- corrected quotient and where it has to go (PH_CHK) ----------------
    // q = Q, 27 bits, in [2^25, 2^27). The rounder wants the kept bits in [26:3]:
    //   normal result    : shift left by nrm (0 or 1)
    //   subnormal result : shift right by d = 1 - E0
    // Both are done as "pre-shift left by nrm (normal results only), then den_k one-bit
    // right steps". Past 31 steps every quotient bit is already sticky, so den_k stops there.
    wire        [26:0] q       = sum[27:1];              // valid in PH_CHK
    wire               nrm     = ~q[26];
    wire               e_neg   = E0[9];
    wire               e_ge1   = (E0 >= 10'sd1);
    wire               e_big   = (E0 >= 10'sd2);
    wire               e_far   = (E0 < -10'sd30);        // shift of more than 31: clamp to 31
    wire               e_zero  = ~e_ge1 & ~e_neg;
    wire               e_one   = e_ge1 & ~e_big;
    wire               pre_l   = e_big & nrm;
    wire        [27:0] q_pre   = pre_l ? sum[27:0] : {1'b0, sum[27:1]};
    wire        [4:0]  k_sh    = ~E0[4:0] + 5'd2;                      // 1 - E0, mod 32
    wire        [5:0]  den_k   = e_ge1 ? 6'd0 : e_far ? 6'd31 : {1'b0, k_sh};

    // tininess after rounding: would the normalised quotient round up to the next binade?
    wire               all_u   = &q[25:3];
    wire               rnd_u   = nrm ? q[1] : q[2];
    wire               low_u   = nrm ? q[0] : (|q[1:0]);
    wire               lsb_u   = nrm ? q[2] : q[3];
    wire               carry_u = rnd_u & (low_u | rem_nz | lsb_u) & all_u & (nrm ? q[2] : 1'b1);
    wire               tiny    = e_neg | (e_zero & nrm) | ((nrm ? e_one : e_zero) & ~carry_u);

    always @(posedge clk) begin
        if (!run) begin                                  // idle: keep the first multiply set up
            H <= 29'd0;
            M <= pow2 ? {m_s, 4'd0} : {seed(m_s[22:19]), 22'd0};      // q = ma exactly, or x0
            if (start)                                   // L holds the last result until then
                L <= special_n ? res_spec : {2'b01, resid[29:26], 2'b00, m_s};
        end else if (cnt != 6'd0) begin                  // multiply step (or plain shift in PH_DEN)
            H <= sum[29:1];
            L <= {sum[0], L[31:1]};
        end else if (ph == PH_CHK) begin                 // hand the quotient to the shifter
            H <= {1'b0, q_pre};
            L <= 32'd0;
            nrm_r  <= nrm;
            tiny_r <= tiny;
        end else if (ph == PH_DEN) begin                 // done: L keeps the packed result
            L <= res_fin;
        end else if (ph != PH_REM) begin                 // phase boundary: load the next multiply
            // L[31:26] is only scanned in the 32-step phases, so it always loads {01, e[29:26]}.
            L <= {2'b01, resid[29:26], to_xt ? resid[25:0] : {2'b00, m_s}};
            // PH_QUO starts at 2^25 (+1 at the quotient LSB); PH_REM starts at 2 * (2^24 - 1)
            H <= {3'd0, x_end, {24{q_now}}, 1'b0};
            if (!to_xt)
                M <= {H[25:0], L[31], L[30] & ~q_now};   // new x, or 2*q1
        end
    end

    // ---------------- remainder sign and self-check ----------------
    // Valid at the end of PH_REM (and still in PH_CHK):
    //   {H, L[31:6]} = 2 * (q1*mb + 2^24 - 1) = 2 * (ma*2^26 + (2^24 - 1 - r)),   r = ma*2^26 - q1*mb.
    // For a correct q1, r is in [-mb, mb), so 2^24 - 1 - r fits in 25 bits and the bits
    // above it are exactly 2*ma. Bit 24 of it is the sign of r; the low 24 bits hold
    // ~r (r >= 0) or -r - 1 (r < 0). m_s is mb at the end of PH_REM and ma in PH_CHK.
    wire        r_sgn  = L[31];
    assign      r_mag  = L[30:7] ^ {24{~r_sgn}};         // r, or -r - 1
    wire [24:0] r_cmp  = sum[24:0];                      // mb - r_mag - 1 on the main adder; carry out = (r_mag < mb)
    wire        r_edge = ~|r_cmp[23:0];                  // r_mag == mb - 1, i.e. r == -mb when negative
    wire        hi_ok  = (H == {4'd0, m_s, 1'b0});

    // ---------------- round (end of PH_DEN) ----------------
    // H[26:0] is the quotient in place, L holds every bit shifted out below it.
    wire               subn    = ~e_ge1 | (e_one & nrm_r);            // result exponent field is 0 (or 1 after carry)
    wire               rnd     = H[2];
    wire               stk     = (|H[1:0]) | rem_nz | (|L);
    wire               inexact = rnd | stk;
    assign             rnd_inc = fin & rnd & (stk | H[3]);
    wire        [24:0] keep2   = sum[27:3];              // kept bits after rounding, valid in fin
    // normal result: exponent = E0 - nrm + carry
    wire signed [9:0]  Er      = E0 + $signed({{9{nrm_r & ~keep2[24]}}, nrm_r ^ keep2[24]});
    wire        [7:0]  e_fld   = subn ? {7'd0, keep2[23]} : Er[7:0];
    wire        [22:0] frac    = keep2[22:0];            // a rounding carry leaves all of these zero
    wire               ovf     = ~subn & (Er >= 10'sd255);
    wire               unf     = tiny_r & inexact;
    wire        [31:0] res_fin = ovf ? {s, 31'h7F800000} : {s, e_fld, frac};
    wire        [31:0] res_spec = nan_n ? 32'h7FC00000 :
                                  inf_n ? {s, 31'h7F800000} : {s, 31'd0};
    assign             result  = L;

    // ---------------- control ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run    <= 1'b0;
            ph     <= 4'd0;
            cnt    <= 6'd0;
            flags  <= 5'd0;
            fault  <= 1'b0;
            r_neg  <= 1'b0;
            rem_nz <= 1'b0;
            done   <= 1'b0;
        end else begin
            done <= 1'b0;
            if (!run) begin
                if (start) begin
                    fault  <= 1'b0;
                    r_neg  <= 1'b0;
                    rem_nz <= 1'b0;
                    if (special_n) begin
                        flags  <= {invalid_n, divz_n, 3'b000};
                        done   <= 1'b1;
                    end else begin
                        run <= 1'b1;
                        ph  <= pow2 ? PH_CHK : PH_BX0;    // early-out skips every multiply
                        cnt <= pow2 ? 6'd0 : 6'd26;
                    end
                end
            end else if (cnt != 6'd0) begin
                cnt <= cnt - 6'd1;
            end else begin
                case (ph)
                    PH_REM: begin                         // m_s = mb
                        r_neg  <= r_sgn;
                        rem_nz <= r_sgn ? ~r_edge : (|r_mag);
                        fault  <= ~r_cmp[24];
                        ph     <= PH_CHK;
                    end
                    PH_CHK: begin                         // m_s = ma
                        if (!pow2 && !hi_ok)
                            fault <= 1'b1;
                        ph  <= PH_DEN;
                        cnt <= den_k;
                    end
                    PH_DEN: begin
                        flags <= ovf ? 5'b00101 : {3'b000, unf, inexact};
                        done <= 1'b1;
                        run  <= 1'b0;
                    end
                    default: begin
                        ph  <= ph + 4'd1;
                        cnt <= to_xt ? 6'd32 : 6'd26;
                    end
                endcase
            end
        end
    end

`ifdef FP32_DEBUG
    // Simulation-only named copies of the values that otherwise exist only as
    // fields of {H, L} at a phase boundary.
    reg [27:0] dbg_x0, dbg_x1, dbg_x2, dbg_x3;
    reg [26:0] dbg_q1;
    always @(posedge clk) begin
        if (!run)
            dbg_x0 <= {seed(m_s[22:19]), 22'd0};   // meaningful on the start clock
        else if (cnt == 6'd0) begin
            if (ph == PH_XT0) dbg_x1 <= {H[25:0], L[31:30]};
            if (ph == PH_XT1) dbg_x2 <= {H[25:0], L[31:30]};
            if (ph == PH_XT2) dbg_x3 <= {H[25:0], L[31:30]};
            if (ph == PH_QUO) dbg_q1 <= {H[25:0], L[31]};
        end
    end
`endif

endmodule
