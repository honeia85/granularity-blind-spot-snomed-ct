"""Resampling intervals of the main analysis of the fixed panel.

1. Document resampling (primary interval). The 245 source documents are indexed in the sorted order of their
   labels. Each of 10,000 replicates draws 245 document positions with replacement (random.Random(12345);
   randrange) and applies each document's multiplicity unchanged to all of its mentions, in every benchmark,
   system and candidate space. Per space: the Mantel-Haenszel risk ratio over the 14 cells, and the equal-cell
   geometric means of the credit-gap, error-incidence and conditional-proximity ratios (undefined in a
   replicate if any cell ratio is undefined or not positive). Cells enter the sums in the sorted order of
   '<benchmark>_<space>_<system>' (lower-case benchmark). Interval: sorted values at 0-based positions
   floor(0.025 B) and floor(0.975 B), reported only when all replicates are defined.
2. Mention resampling within benchmarks. Each of 10,000 replicates draws, for each
   benchmark in turn, as many mention positions as the benchmark has, with replacement (random.Random(12345)),
   and applies them to every system. Mantel-Haenszel risk ratio and odds ratio and the crude collapsed odds
   ratio over the 14 gold-inclusive cells; interval positions as above.
3. Credit-gap ratio (B = 2,000; seed 20260805). Per cell, EXACT and NARROW mentions are resampled
   independently (all EXACT draws, then all NARROW draws, cells in benchmark and system order, one
   random.Random generator). For the geometric mean over the 14 cells, a second generator with the same seed
   draws EXACT and then NARROW positions once per benchmark and applies them to every system; replicates with
   any undefined ratio are dropped. Percentile intervals with linear interpolation.
4. Additive credit gap, Delta G = gap(NARROW) - gap(EXACT) (B = 10,000; random.Random seeded with 20260805), in both
   candidate spaces. Per cell, EXACT and NARROW mentions are resampled independently (in each replicate all EXACT draws,
   then all NARROW draws; cells in space, benchmark and system order; one generator per space). Percentile intervals
   with linear interpolation; the number of cells whose interval excludes zero is reported per space.
5. Fixed-stratum item bootstrap (B = 10,000; numpy-compatible PCG64 generator with seed 42; see pcg64.py).
   For each benchmark in turn, a 10,000 x n_EXACT array and then a 10,000 x n_NARROW array of positions are
   drawn (row by row) and applied to every system. Statistics: the NARROW-minus-EXACT difference of mean
   Wu-Palmer among errors, averaged over the cells of a benchmark and over all 14 cells; and the
   NARROW-minus-EXACT difference of the is-a uplift, mean(is-a) - mean(exact), per cell. Percentile intervals
   with linear interpolation.
Sums of Wu-Palmer scores are exact sums of 1/10000 units; vectors of counts are added as packed integers.
"""
import math
import random

import pcg64
from common import BENCHMARKS
from main_scores import SYSTEMS, WUP_SCALE, load_main

FIELD = 40
MASK = (1 << FIELD) - 1


def pack(values):
    v = 0
    for k, x in enumerate(values):
        assert 0 <= x < (1 << (FIELD - 8))
        v |= x << (FIELD * k)
    return v


def unpack(v, count):
    return [(v >> (FIELD * k)) & MASK for k in range(count)]


def draws_below(getrandbits, n, count):
    """count draws of random.Random.randrange(n) (getrandbits rejection, as in CPython)."""
    k = n.bit_length()
    out = []
    append = out.append
    for _ in range(count):
        r = getrandbits(k)
        while r >= n:
            r = getrandbits(k)
        append(r)
    return out


def fixed_positions(values, B):
    s = sorted(values)
    return s[int(math.floor(0.025 * B))], s[int(math.floor(0.975 * B))]


def linear_interval(values):
    v = [x for x in values if x is not None and math.isfinite(x)]
    return pcg64.quantile_linear(v, 0.025), pcg64.quantile_linear(v, 0.975)


def ratio_metrics(vectors):
    """vectors: per cell [nE, eE, gE, nN, eN, gN] (g in Wu-Palmer units). Returns MH, G, E, Q (None if undefined)."""
    num = den = 0.0
    lg = le = lq = 0.0
    good = True
    for nE, eE, gE, nN, eN, gN in vectors:
        if nE and nN:
            num += eN * nE / (nE + nN)
            den += eE * nN / (nE + nN)
        try:
            er = (eN / nN) / (eE / nE)
            gr = (gN / nN) / (gE / nE)
            qr = (gN / eN) / (gE / eE)
        except ZeroDivisionError:
            good = False
            continue
        if er > 0 and gr > 0 and qr > 0:
            le += math.log(er)
            lg += math.log(gr)
            lq += math.log(qr)
        else:
            good = False
    k = len(vectors)
    mh = num / den if den else None
    if not good:
        return mh, None, None, None
    return mh, math.exp(lg / k), math.exp(le / k), math.exp(lq / k)


def document_resampling(by_bench, cells, out, B=10000, seed=12345):
    docs = sorted({r['document_id'] for b in BENCHMARKS for r in by_bench[b]})
    D = len(docs)
    dpos = {d: k for k, d in enumerate(docs)}
    keys = sorted(cells, key=lambda k: '%s_%s_%s_per_mention.csv' % (k[1].lower(), k[0], k[2]))
    vec = [[0] * (6 * len(keys)) for _ in range(D)]
    for j, key in enumerate(keys):
        b = key[1]
        for item, row in zip(by_bench[b], cells[key]):
            z = vec[dpos[item['document_id']]]
            o = 6 * j + (3 if row[0] else 0)
            z[o] += 1
            z[o + 1] += 1 - row[1]
            z[o + 2] += row[3] - row[1] * WUP_SCALE
    packed = [pack(v) for v in vec]
    spaces = {s: [j for j, k in enumerate(keys) if k[0] == s] for s in ('oracle', 'train')}

    def metrics(total):
        f = unpack(total, 6 * len(keys))
        res = {}
        for s, js in spaces.items():
            mh, g, e, q = ratio_metrics([f[6 * j:6 * j + 6] for j in js])
            res[s] = (mh, g, e, q)
        return res

    observed = metrics(sum(packed))
    rng = random.Random(seed)
    getrandbits = rng.getrandbits
    reps = {s: ([], [], [], []) for s in spaces}
    for _ in range(B):
        cnt = [0] * D
        for d in draws_below(getrandbits, D, D):
            cnt[d] += 1
        total = 0
        for d in range(D):
            if cnt[d]:
                total += cnt[d] * packed[d]
        m = metrics(total)
        for s in spaces:
            for k in range(4):
                reps[s][k].append(m[s][k])
    out['document_resampling_documents'] = D
    out['document_resampling_replicates'] = B
    out['document_resampling_seed'] = seed
    for s in spaces:
        for k, name in enumerate(('mh_risk_ratio', 'geometric_mean_gap_ratio', 'geometric_mean_incidence_ratio',
                                  'geometric_mean_proximity_ratio')):
            vals = reps[s][k]
            valid = [v for v in vals if v is not None and math.isfinite(v)]
            pre = 'document_resampling_%s_%s' % (s, name)
            out[pre] = observed[s][k]
            out[pre + '_undefined'] = B - len(valid)
            if len(valid) == B:
                out[pre + '_ci_low'], out[pre + '_ci_high'] = fixed_positions(valid, B)
            else:
                out[pre + '_ci_low'] = out[pre + '_ci_high'] = None


def mention_resampling(by_bench, cells, out, B=10000, seed=12345):
    """Mantel-Haenszel risk and odds ratios with mentions resampled within benchmarks (shared across systems)."""
    per_bench = []
    for b in BENCHMARKS:
        systems = [s for s in SYSTEMS if ('oracle', b, s) in cells]
        rows = [cells[('oracle', b, s)] for s in systems]
        packed = []
        for i in range(len(by_bench[b])):
            narrow = rows[0][i][0]
            v = [narrow, 1 - narrow]
            for r in rows:
                err = 1 - r[i][1]
                v += [err * narrow, err * (1 - narrow)]
            packed.append(pack(v))
        per_bench.append((b, len(systems), packed))

    def estimates(totals):
        rr_num = rr_den = or_num = or_den = 0.0
        A = Bc = C = Dd = 0
        logs = []
        for (b, k, _), t in zip(per_bench, totals):
            f = unpack(t, 2 + 2 * k)
            n1, n0 = f[0], f[1]
            for j in range(k):
                a, c = f[2 + 2 * j], f[3 + 2 * j]
                if n1 and n0:
                    rr_num += a * n0 / (n1 + n0)
                    rr_den += c * n1 / (n1 + n0)
                b_ok, d_ok = n1 - a, n0 - c
                n = n1 + n0
                if n:
                    or_num += a * d_ok / n
                    or_den += b_ok * c / n
                A, Bc, C, Dd = A + a, Bc + b_ok, C + c, Dd + d_ok
                if a and c and n1 and n0:
                    logs.append(math.log((a / n1) / (c / n0)))
        rr = rr_num / rr_den if rr_den else float('nan')
        orr = or_num / or_den if or_den else float('nan')
        crude = (A * Dd) / (Bc * C) if Bc * C else float('nan')
        gm = math.exp(math.fsum(logs) / len(logs))
        return rr, orr, crude, gm

    observed = estimates([sum(p) for _, _, p in per_bench])
    rng = random.Random(seed)
    getrandbits = rng.getrandbits
    reps = ([], [], [], [])
    for _ in range(B):
        totals = []
        for _, _, packed in per_bench:
            n = len(packed)
            totals.append(sum(packed[i] for i in draws_below(getrandbits, n, n)))
        for k, v in enumerate(estimates(totals)):
            reps[k].append(v)
    out['mention_resampling_replicates'] = B
    out['mention_resampling_seed'] = seed
    for k, name in enumerate(('mh_risk_ratio', 'mh_odds_ratio', 'crude_odds_ratio', 'geometric_mean_incidence_ratio')):
        pre = 'mention_resampling_' + name
        out[pre] = observed[k]
        out[pre + '_ci_low'], out[pre + '_ci_high'] = fixed_positions(reps[k], B)


def gap_units(rows):
    return [r[3] - r[1] * WUP_SCALE for r in rows]


def gap_ratio_intervals(by_bench, cells, out, B=2000, seed=20260805):
    rng = random.Random(seed)
    getrandbits = rng.getrandbits
    ratios = {}
    excluding = 0
    for b in BENCHMARKS:
        for s in SYSTEMS:
            key = ('oracle', b, s)
            if key not in cells:
                continue
            ex = gap_units([r for r in cells[key] if r[0] == 0])
            na = gap_units([r for r in cells[key] if r[0] == 1])
            ne, nn = len(ex), len(na)
            point = (sum(na) / nn) / (sum(ex) / ne)
            ratios[(b, s)] = point
            draws = []
            for _ in range(B):
                ge = sum(map(ex.__getitem__, draws_below(getrandbits, ne, ne)))
                gn = sum(map(na.__getitem__, draws_below(getrandbits, nn, nn)))
                draws.append((gn / nn) / (ge / ne) if ge > 0 else float('nan'))
            lo, hi = linear_interval(draws)
            pre = 'gap_ratio_%s_%s' % (b, s)
            out[pre] = point
            out[pre + '_ci_low'], out[pre + '_ci_high'] = lo, hi
            excluding += lo > 1.0
    out['gap_ratio_cells_ci_excluding_1'] = excluding
    out['gap_ratio_replicates'] = B
    out['gap_ratio_seed'] = seed
    # geometric mean with indices shared across systems within each benchmark
    joint = random.Random(seed)
    getrandbits = joint.getrandbits
    per_bench = []
    for b in BENCHMARKS:
        systems = [s for s in SYSTEMS if ('oracle', b, s) in cells]
        rows = [cells[('oracle', b, s)] for s in systems]
        e_idx = [i for i, r in enumerate(rows[0]) if r[0] == 0]
        n_idx = [i for i, r in enumerate(rows[0]) if r[0] == 1]
        pk = [pack([r[i][3] - r[i][1] * WUP_SCALE for r in rows]) for i in range(len(rows[0]))]
        per_bench.append((len(systems), e_idx, n_idx, pk))
    geo = []
    for _ in range(B):
        vals = []
        for k, e_idx, n_idx, pk in per_bench:
            se = draws_below(getrandbits, len(e_idx), len(e_idx))
            sn = draws_below(getrandbits, len(n_idx), len(n_idx))
            fe = unpack(sum(pk[e_idx[i]] for i in se), k)
            fn = unpack(sum(pk[n_idx[i]] for i in sn), k)
            for j in range(k):
                ge, gn = fe[j] / len(e_idx), fn[j] / len(n_idx)
                if ge > 0:
                    vals.append(gn / ge)
        if len(vals) == len(ratios) and all(v > 0 for v in vals):
            geo.append(math.exp(math.fsum(math.log(v) for v in vals) / len(vals)))
    out['gap_ratio_geometric_mean'] = math.exp(math.fsum(math.log(v) for v in ratios.values()) / len(ratios))
    out['gap_ratio_geometric_mean_ci_low'], out['gap_ratio_geometric_mean_ci_high'] = linear_interval(geo)


def additive_gap_intervals(by_bench, cells, out, B=10000, seed=20260805):
    for space in ('oracle', 'train'):
        rng = random.Random(seed)
        getrandbits = rng.getrandbits
        excluding = 0
        for b in BENCHMARKS:
            for s in SYSTEMS:
                key = (space, b, s)
                if key not in cells:
                    continue
                ex = gap_units([r for r in cells[key] if r[0] == 0])
                na = gap_units([r for r in cells[key] if r[0] == 1])
                ne, nn = len(ex), len(na)
                draws = []
                for _ in range(B):
                    ge = sum(map(ex.__getitem__, draws_below(getrandbits, ne, ne)))
                    gn = sum(map(na.__getitem__, draws_below(getrandbits, nn, nn)))
                    draws.append((gn / nn - ge / ne) / WUP_SCALE)
                lo, hi = linear_interval(draws)
                pre = 'gap_difference_%s_%s_%s' % (space, b, s)
                out[pre] = (sum(na) / nn - sum(ex) / ne) / WUP_SCALE
                out[pre + '_ci_low'], out[pre + '_ci_high'] = lo, hi
                excluding += lo > 0 or hi < 0
        out['gap_difference_%s_cells_ci_excluding_0' % space] = excluding
    out['gap_difference_replicates'] = B
    out['gap_difference_seed'] = seed


def fixed_stratum_bootstrap(by_bench, cells, out, B=10000, seed=42):
    rng = pcg64.Generator(seed)
    all_cp, all_points = [], []
    uplift_excl = 0
    for b in BENCHMARKS:
        systems = [s for s in SYSTEMS if ('oracle', b, s) in cells]
        rows = [cells[('oracle', b, s)] for s in systems]
        k = len(systems)
        strata = []
        for narrow in (0, 1):
            idx = [i for i, r in enumerate(rows[0]) if r[0] == narrow]
            pk = []
            for i in idx:
                v = []
                for r in rows:
                    x = r[i]
                    v += [x[3] * (1 - x[1]), 1 - x[1], x[2], x[1]]
                pk.append(pack(v))
            strata.append((idx, pk))
        (e_idx, e_pk), (n_idx, n_pk) = strata
        ne, nn = len(e_idx), len(n_idx)
        e_draw = rng.integers(ne, B * ne)
        n_draw = rng.integers(nn, B * nn)

        def stats(fe, fn):
            cp, up = [], []
            for j in range(k):
                we, ee, ie, xe = fe[4 * j:4 * j + 4]
                wn, en, i_n, xn = fn[4 * j:4 * j + 4]
                cp.append((wn / WUP_SCALE / en if en else float('nan')) - (we / WUP_SCALE / ee if ee else float('nan')))
                up.append((i_n - xn) / nn - (ie - xe) / ne)
            return cp, up

        cp0, up0 = stats(unpack(sum(e_pk), 4 * k), unpack(sum(n_pk), 4 * k))
        cp_reps = [[] for _ in range(k)]
        up_reps = [[] for _ in range(k)]
        for rep in range(B):
            fe = unpack(sum(e_pk[i] for i in e_draw[rep * ne:(rep + 1) * ne]), 4 * k)
            fn = unpack(sum(n_pk[i] for i in n_draw[rep * nn:(rep + 1) * nn]), 4 * k)
            cp, up = stats(fe, fn)
            for j in range(k):
                cp_reps[j].append(cp[j])
                up_reps[j].append(up[j])
        bench_boot = [math.fsum(cp_reps[j][rep] for j in range(k)) / k for rep in range(B)]
        out['conditional_wup_difference_%s' % b] = math.fsum(cp0) / k
        out['conditional_wup_difference_%s_ci_low' % b], out['conditional_wup_difference_%s_ci_high' % b] = linear_interval(bench_boot)
        all_cp.extend(cp_reps)
        all_points.extend(cp0)
        for j, s in enumerate(systems):
            pre = 'uplift_difference_%s_%s' % (b, s)
            out[pre] = up0[j]
            lo, hi = linear_interval(up_reps[j])
            out[pre + '_ci_low'], out[pre + '_ci_high'] = lo, hi
            uplift_excl += lo > 0 or hi < 0
    overall = [math.fsum(c[rep] for c in all_cp) / len(all_cp) for rep in range(B)]
    out['conditional_wup_difference'] = math.fsum(all_points) / len(all_points)
    out['conditional_wup_difference_ci_low'], out['conditional_wup_difference_ci_high'] = linear_interval(overall)
    out['uplift_difference_cells_ci_excluding_0'] = uplift_excl
    out['fixed_stratum_replicates'] = B
    out['fixed_stratum_seed'] = seed


def run(data_dir):
    by_bench, cells = load_main(data_dir)
    out = {}
    document_resampling(by_bench, cells, out)
    mention_resampling(by_bench, cells, out)
    gap_ratio_intervals(by_bench, cells, out)
    additive_gap_intervals(by_bench, cells, out)
    fixed_stratum_bootstrap(by_bench, cells, out)
    return out
