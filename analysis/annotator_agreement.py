"""Re-annotation of the granularity label by three clinical annotators on a balanced 150-mention sample.

The sample holds 25 corpus-EXACT and 25 corpus-NARROW mentions from each Spanish benchmark. Each annotator (R1,
R2, R3) labelled every mention EXACT or NARROW, blinded to the corpus label and to system outputs.
Statistics: Cohen's kappa and raw agreement of each annotator with the corpus label (overall and by benchmark);
pairwise Cohen's kappa between annotators; Fleiss' kappa over the three annotators; the majority-vote label and
its kappa and agreement with the corpus label.
Intervals: item bootstrap, 5,000 replicates, numpy-compatible PCG64 generator seeded with 42 anew for every
statistic (see pcg64.py); each replicate draws as many item positions as the statistic has items (items in label
order); replicates in which either rating vector has a single category are skipped for Cohen's kappa; percentile
intervals with linear interpolation.
"""
import math

import pcg64
from common import BENCHMARKS, read_csv

LABELS = ['EXACT', 'NARROW']
RATERS = ['R1', 'R2', 'R3']
SEED = 42
REPLICATES = 5000


def cohen(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = math.fsum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in sorted(set(a) | set(b)))
    return (po - pe) / (1 - pe)


def cohen_interval(a, b):
    rng = pcg64.Generator(SEED)
    n = len(a)
    draws = rng.integers(n, REPLICATES * n)
    boot = []
    for r in range(REPLICATES):
        idx = draws[r * n:(r + 1) * n]
        x = [a[i] for i in idx]
        y = [b[i] for i in idx]
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        boot.append(cohen(x, y))
    return pcg64.quantile_linear(boot, 0.025), pcg64.quantile_linear(boot, 0.975)


def fleiss(counts):
    n = len(counts)
    m = sum(counts[0])
    p = [sum(c[j] for c in counts) / (n * m) for j in range(len(LABELS))]
    P = [(sum(x * x for x in c) - m) / (m * (m - 1)) for c in counts]
    pbar = math.fsum(P) / n
    pe = math.fsum(x * x for x in p)
    return (pbar - pe) / (1 - pe)


def fleiss_interval(counts):
    rng = pcg64.Generator(SEED)
    n = len(counts)
    draws = rng.integers(n, REPLICATES * n)
    boot = []
    for r in range(REPLICATES):
        try:
            boot.append(fleiss([counts[i] for i in draws[r * n:(r + 1) * n]]))
        except ZeroDivisionError:
            continue
    return pcg64.quantile_linear(boot, 0.025), pcg64.quantile_linear(boot, 0.975)


def add_kappa(out, name, ref, lab):
    out[name + '_kappa'] = cohen(ref, lab)
    out[name + '_kappa_ci_low'], out[name + '_kappa_ci_high'] = cohen_interval(ref, lab)
    out[name + '_agreement'] = sum(1 for x, y in zip(ref, lab) if x == y) / len(ref)
    out[name + '_agreements'] = sum(1 for x, y in zip(ref, lab) if x == y)


def run(data_dir):
    rows = read_csv(data_dir, 'reannotation_labels.csv')
    out = {'items': len(rows), 'replicates': REPLICATES, 'seed': SEED}
    ref = [r['corpus_label'] for r in rows]
    out['corpus_narrow'] = sum(1 for x in ref if x == 'NARROW')
    for b in BENCHMARKS:
        out['items_' + b] = sum(1 for r in rows if r['benchmark'] == b)
    for k in RATERS:
        lab = [r[k] for r in rows]
        out[k + '_narrow'] = sum(1 for x in lab if x == 'NARROW')
        add_kappa(out, k, ref, lab)
        for b in BENCHMARKS:
            sub = [r for r in rows if r['benchmark'] == b]
            add_kappa(out, '%s_%s' % (k, b), [r['corpus_label'] for r in sub], [r[k] for r in sub])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = RATERS[i], RATERS[j]
        x = [r[a] for r in rows]
        y = [r[b] for r in rows]
        out['%s_%s_kappa' % (a, b)] = cohen(x, y)
        out['%s_%s_kappa_ci_low' % (a, b)], out['%s_%s_kappa_ci_high' % (a, b)] = cohen_interval(x, y)
    counts = [[sum(1 for k in RATERS if r[k] == c) for c in LABELS] for r in rows]
    out['fleiss_kappa'] = fleiss(counts)
    out['fleiss_kappa_ci_low'], out['fleiss_kappa_ci_high'] = fleiss_interval(counts)
    majority = ['NARROW' if c[1] >= 2 else 'EXACT' for c in counts]
    add_kappa(out, 'majority', ref, majority)
    for label in LABELS:
        out['majority_confirms_corpus_' + label] = sum(1 for x, y in zip(ref, majority) if x == y == label)
        out['corpus_' + label] = sum(1 for x in ref if x == label)
    return out
