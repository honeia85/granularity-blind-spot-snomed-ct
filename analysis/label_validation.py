"""Label validation by two clinicians: agreement, misclassification of the corpus labels and corrected risk ratio.

Two clinicians independently judged 300 fixed-panel mentions (50 per benchmark and corpus label) as 'same'
(same level as the assigned concept; reference EXACT), 'narrower' (more specific than the assigned concept;
reference NARROW) or 'hold'. The primary reference is the concordant non-held judgment. For each benchmark,
p is the share of corpus EXACT mentions with reference NARROW and q the share of corpus NARROW mentions with
reference EXACT (Wilson 95% intervals). Pooled p and q weight the benchmarks by the panel's corpus EXACT and
NARROW counts, with a Wilson interval on the effective sample size of the weighted estimate.

Forward correction: the panel Mantel-Haenszel error risk ratio over the gold-inclusive cells is computed on expected
cell tables in which shares p of corpus EXACT and q of corpus NARROW mentions, with the same shares of their
errors, move to the other group. Inverse correction: each cell is solved for the error rates of reference
EXACT and NARROW mentions, which are bounded to [0, 1], and the ratio is computed from these rates. Alternative reference
definitions assign held mentions to 'same' (S1) or 'narrower' (S2), discordant mentions to the corpus label (S3)
or against it (S4), or use one clinician alone.

Intervals: judged mentions are resampled within benchmark-by-label strata (10,000 replicates), and jointly with
panel documents (2,000 replicates; documents are indexed by validation_position in panel_documents.csv); one random.Random generator with the seed below; percentile intervals.
The kappa interval resamples judged mentions within the same strata (10,000 replicates, generator seeded anew).
"""
import collections
import math
import random

from common import BENCHMARKS, kappa, load_panel, percentile_interval, plain_sum, read_csv, wilson, wilson_n

SEED = 20261111
REPLICATES_JUDGMENTS = 10000
REPLICATES_JOINT = 2000
REFERENCE = {'same': 'EXACT', 'narrower': 'NARROW'}
DEFINITIONS = ['primary', 'S1', 'S2', 'S3', 'S4', 'rater_A', 'rater_B']


def tables_by_document(items, cells, data_dir):
    """Per cell: (benchmark, {document position: [nE, eE, nN, eN]}), documents at their validation_position."""
    position = {r['document_id']: int(r['validation_position']) for r in read_csv(data_dir, 'panel_documents.csv')}
    out = []
    for c in cells:
        agg = collections.defaultdict(lambda: [0, 0, 0, 0])
        for i, e in zip(c['idx'], c['err']):
            z = agg[position[items[i]['document_id']]]
            if items[i]['label'] == 'NARROW':
                z[2] += 1
                z[3] += e
            else:
                z[0] += 1
                z[1] += e
        out.append((c['benchmark'], dict(agg)))
    return out, len(position)


def totals(per_doc, weight=None):
    out = []
    for b, agg in per_doc:
        t = [0, 0, 0, 0]
        for d, z in agg.items():
            k = 1 if weight is None else weight.get(d, 0)
            if k:
                for j in range(4):
                    t[j] += k * z[j]
        out.append((b, t))
    return out


def forward_ratio(tables, pq):
    num = den = 0.0
    for b, (nE, eE, nN, eN) in tables:
        p, q = pq[b]
        n = nE + nN
        xN, xE = (1 - q) * nN + p * nE, q * nN + (1 - p) * nE
        aN, aE = (1 - q) * eN + p * eE, q * eN + (1 - p) * eE
        if xE > 0 and xN > 0 and n > 0:
            num += aN * xE / n
            den += aE * xN / n
    return num / den if den else float('nan')


def inverse_ratio(tables, pq):
    num = den = 0.0
    bounded = 0
    for b, (nE, eE, nN, eN) in tables:
        p, q = pq[b]
        if not (math.isfinite(p) and math.isfinite(q)) or p + q >= 1 or nE == 0 or nN == 0:
            return float('nan'), bounded
        rE, rN = eE / nE, eN / nN
        det = 1 - p - q
        aE = ((1 - q) * rE - p * rN) / det
        aN = ((1 - p) * rN - q * rE) / det
        if not (0 <= aE <= 1 and 0 <= aN <= 1):
            bounded += 1
        aE, aN = min(1, max(0, aE)), min(1, max(0, aN))
        mE, mN = (1 - p) * nE + q * nN, p * nE + (1 - q) * nN
        if mE > 0 and mN > 0:
            num += aN * mN * mE / (mE + mN)
            den += aE * mE * mN / (mE + mN)
    return (num / den if den else float('nan')), bounded


def reference(items, rule):
    hold = {'S1': 'EXACT', 'S2': 'NARROW'}.get(rule)
    out = {}
    for it in items:
        ra = REFERENCE.get(it['A']) or (hold if it['A'] == 'hold' else None)
        rb = REFERENCE.get(it['B']) or (hold if it['B'] == 'hold' else None)
        if rule == 'rater_A':
            out[it['id']] = ra
        elif rule == 'rater_B':
            out[it['id']] = rb
        elif ra is not None and ra == rb:
            out[it['id']] = ra
        elif ra is not None and rb is not None and rule == 'S3':
            out[it['id']] = it['label']
        elif ra is not None and rb is not None and rule == 'S4':
            out[it['id']] = 'NARROW' if it['label'] == 'EXACT' else 'EXACT'
        else:
            out[it['id']] = None
    return out


def estimate(items, ref, counts):
    by = {}
    for b in BENCHMARKS:
        e = [it for it in items if it['benchmark'] == b and it['label'] == 'EXACT' and ref[it['id']]]
        n = [it for it in items if it['benchmark'] == b and it['label'] == 'NARROW' and ref[it['id']]]
        kp = sum(ref[it['id']] == 'NARROW' for it in e)
        kq = sum(ref[it['id']] == 'EXACT' for it in n)
        by[b] = {'p': wilson(kp, len(e)), 'p_k': kp, 'p_n': len(e), 'q': wilson(kq, len(n)), 'q_k': kq, 'q_n': len(n)}
    pooled = {}
    for key, lab in (('p', 'EXACT'), ('q', 'NARROW')):
        tot = sum(counts[b][lab] for b in BENCHMARKS)
        w = {b: counts[b][lab] / tot for b in BENCHMARKS}
        est = plain_sum(w[b] * by[b][key][0] for b in BENCHMARKS)
        var = plain_sum(w[b] ** 2 * by[b][key][0] * (1 - by[b][key][0]) / by[b][key + '_n'] for b in BENCHMARKS)
        neff = est * (1 - est) / var if var > 0 and 0 < est < 1 else sum(by[b][key + '_n'] for b in BENCHMARKS)
        pooled[key] = wilson_n(est, neff)
    return by, pooled


def resample(items, ref, per_doc, n_docs, counts, rng):
    strata = collections.defaultdict(list)
    for it in items:
        strata[(it['benchmark'], it['label'])].append(it)
    base = totals(per_doc)

    def draw():
        pq = {}
        for b in BENCHMARKS:
            v = []
            for lab, other in (('EXACT', 'NARROW'), ('NARROW', 'EXACT')):
                s = strata[(b, lab)]
                use = [ref[x['id']] for x in (s[rng.randrange(len(s))] for _ in s) if ref[x['id']]]
                v.append(sum(r == other for r in use) / len(use) if use else float('nan'))
            pq[b] = tuple(v)
        pooled = []
        for k, lab in ((0, 'EXACT'), (1, 'NARROW')):
            tot = sum(counts[b][lab] for b in BENCHMARKS)
            pooled.append(plain_sum(counts[b][lab] / tot * pq[b][k] for b in BENCHMARKS))
        return pq, pooled

    fw, fw_pooled, inv, pp, qq = [], [], [], [], []
    for _ in range(REPLICATES_JUDGMENTS):
        pq, pooled = draw()
        fw.append(forward_ratio(base, pq))
        inv.append(inverse_ratio(base, pq)[0])
        fw_pooled.append(forward_ratio(base, {b: tuple(pooled) for b in BENCHMARKS}))
        pp.append(pooled[0])
        qq.append(pooled[1])
    fw2, inv2 = [], []
    for _ in range(REPLICATES_JOINT):
        pq, _ = draw()
        w = collections.Counter(rng.randrange(n_docs) for _ in range(n_docs))
        tables = totals(per_doc, w)
        fw2.append(forward_ratio(tables, pq))
        inv2.append(inverse_ratio(tables, pq)[0])
    undefined = sum(1 for v in fw + fw2 + inv + inv2 + fw_pooled if not math.isfinite(v))
    return {'forward_bootstrap': percentile_interval(fw), 'forward_joint_bootstrap': percentile_interval(fw2),
            'forward_pooled_bootstrap': percentile_interval(fw_pooled), 'inverse_bootstrap': percentile_interval(inv),
            'inverse_joint_bootstrap': percentile_interval(inv2), 'pooled_p_bootstrap': percentile_interval(pp),
            'pooled_q_bootstrap': percentile_interval(qq), 'undefined': undefined}


def run(data_dir):
    panel_items, _, cells = load_panel(data_dir)
    per_doc, n_docs = tables_by_document(panel_items, cells, data_dir)
    counts = {b: {'EXACT': 0, 'NARROW': 0} for b in BENCHMARKS}
    for it in panel_items:
        counts[it['benchmark']][it['label']] += 1
    items = {}
    for r in read_csv(data_dir, 'label_validation_judgments.csv'):
        it = items.setdefault(r['item_id'], {'id': r['item_id'], 'benchmark': r['benchmark'], 'label': r['corpus_label']})
        it[r['rater']] = r['choice']
    items = list(items.values())
    base = totals(per_doc)
    out = {}
    out['items'] = len(items)
    for b in BENCHMARKS:
        for lab in ('EXACT', 'NARROW'):
            out['panel_%s_%s' % (lab.lower(), b)] = counts[b][lab]
    out['uncorrected_ratio'] = forward_ratio(base, {b: (0.0, 0.0) for b in BENCHMARKS})
    for r in ('A', 'B'):
        c = collections.Counter(it[r] for it in items)
        out['rater_%s_judgments' % r] = sum(c.values())
        for k in ('same', 'narrower', 'hold'):
            out['rater_%s_%s' % (r, k)] = c.get(k, 0)

    three = [(it['A'], it['B']) for it in items]
    two = [(a, b) for a, b in three if 'hold' not in (a, b)]
    k3, k2 = kappa(three, ['same', 'narrower', 'hold']), kappa(two, ['same', 'narrower'])
    out['three_category_items'], out['three_category_agreement'], out['three_category_kappa'] = k3['n'], k3['agreement'], k3['kappa']
    out['three_category_agreements'] = sum(a == b for a, b in three)
    out['two_category_items'], out['two_category_agreement'], out['two_category_kappa'] = k2['n'], k2['agreement'], k2['kappa']
    out['both_held'] = sum(1 for a, b in three if a == b == 'hold')
    for b in BENCHMARKS:
        sub = [(it['A'], it['B']) for it in items if it['benchmark'] == b]
        x3, x2 = kappa(sub, ['same', 'narrower', 'hold']), kappa([(u, v) for u, v in sub if 'hold' not in (u, v)], ['same', 'narrower'])
        out['three_category_items_' + b], out['three_category_kappa_' + b] = x3['n'], x3['kappa']
        out['two_category_items_' + b], out['two_category_kappa_' + b] = x2['n'], x2['kappa']

    rng = random.Random(SEED)
    strata = collections.defaultdict(list)
    for it in items:
        strata[(it['benchmark'], it['label'])].append(it)
    ks = []
    for _ in range(REPLICATES_JUDGMENTS):
        pick = [s[rng.randrange(len(s))] for s in strata.values() for _ in s]
        ks.append(kappa([(it['A'], it['B']) for it in pick if 'hold' not in (it['A'], it['B'])], ['same', 'narrower'])['kappa'])
    out['two_category_kappa_ci_low'], out['two_category_kappa_ci_high'] = percentile_interval(ks)

    for rule in DEFINITIONS:
        ref = reference(items, rule)
        by, pooled = estimate(items, ref, counts)
        pq = {b: (by[b]['p'][0], by[b]['q'][0]) for b in BENCHMARKS}
        inv, bounded = inverse_ratio(base, pq)
        pre = rule + '_'
        for key in ('p', 'q'):
            out[pre + 'pooled_' + key], out[pre + 'pooled_%s_ci_low' % key], out[pre + 'pooled_%s_ci_high' % key] = pooled[key]
        out[pre + 'corrected_ratio'] = forward_ratio(base, pq)
        out[pre + 'corrected_ratio_pooled'] = forward_ratio(base, {b: (pooled['p'][0], pooled['q'][0]) for b in BENCHMARKS})
        out[pre + 'inverse_ratio'] = inv
        out[pre + 'inverse_cells_bounded'] = bounded
        if rule != 'primary':
            continue
        for b in BENCHMARKS:
            for key in ('p', 'q'):
                v = by[b]
                out['%s%s_%s' % (pre, key, b)], out['%s%s_%s_ci_low' % (pre, key, b)], out['%s%s_%s_ci_high' % (pre, key, b)] = v[key]
                out['%s%s_%s_count' % (pre, key, b)], out['%s%s_%s_items' % (pre, key, b)] = v[key + '_k'], v[key + '_n']
            for lab in ('EXACT', 'NARROW'):
                sub = [it for it in items if it['benchmark'] == b and it['label'] == lab]
                out['%sstratum_%s_%s_items' % (pre, lab.lower(), b)] = len(sub)
                out['%sstratum_%s_%s_concordant' % (pre, lab.lower(), b)] = sum(1 for it in sub if ref[it['id']])
        for key in ('p', 'q'):
            out['%s%s_count' % (pre, key)] = sum(by[b][key + '_k'] for b in BENCHMARKS)
            out['%s%s_items' % (pre, key)] = sum(by[b][key + '_n'] for b in BENCHMARKS)
        for lab in ('EXACT', 'NARROW'):
            sub = [it for it in items if it['label'] == lab]
            for target in ('EXACT', 'NARROW'):
                out['%scorpus_%s_reference_%s' % (pre, lab.lower(), target.lower())] = sum(1 for it in sub if ref[it['id']] == target)
            out['%sstratum_%s_items' % (pre, lab.lower())] = len(sub)
            out['%sstratum_%s_concordant' % (pre, lab.lower())] = sum(1 for it in sub if ref[it['id']])
        ci = resample(items, ref, per_doc, n_docs, counts, random.Random(SEED))
        out['replicates_judgments'], out['replicates_joint'], out['seed'] = REPLICATES_JUDGMENTS, REPLICATES_JOINT, SEED
        for name, value in ci.items():
            if name != 'undefined':
                out[pre + name + '_ci_low'], out[pre + name + '_ci_high'] = value
        out[pre + 'undefined_replicates'] = ci['undefined']
    alternatives = [out[r + '_corrected_ratio'] for r in DEFINITIONS if r != 'primary']
    out['alternative_definitions_corrected_ratio_min'] = min(alternatives)
    out['alternative_definitions_corrected_ratio_max'] = max(alternatives)
    return out
