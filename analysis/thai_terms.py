"""Thai term-level evaluation: errors by label group, NARROW/EXACT error risk ratio and two bootstrap intervals.

The same computation is applied to the errors of three systems: a lexical character n-gram baseline, SapBERT
retrieval, and SapBERT retrieval with the trailing parenthetical tag removed from the candidate labels.

Terms: the 999 source codes of the mapping table with a single ACCEPTED row whose relationship type is
TARGET_EQUIVALENT (EXACT, 952) or TARGET_BROADER (NARROW, 47), whose Term value is written in Latin letters without
Thai characters, and whose target code passes the SNOMED CT check digit and is not given as a numeric cell of more
than 15 digits. Candidates: the 1,421 distinct target codes of all ACCEPTED rows that pass the same code checks.
Each term is an item; the 852 distinct target codes of the terms form the clusters.

Cluster bootstrap: the reference-concept clusters (rows of thai_clusters.csv in position order) are resampled with
replacement, 20,000 replicates, each a multinomial draw over the clusters from a PCG64 generator seeded with 12345;
all items of a sampled cluster are kept. Item bootstrap: within each label group the items (rows of thai_items.csv
in position order) are resampled with replacement from a generator seeded with 54321; replicates are drawn in
blocks of 500, first 500 EXACT-group draws and then 500 NARROW-group draws. Intervals are 95% percentile
intervals (inverted empirical distribution function). The generator and the sampling order equal those of
numpy.random.default_rng(seed).multinomial, so numpy is not needed (see pcg64.py).
"""
import math

from common import read_csv
from pcg64 import Generator, quantile_inverted_cdf

REPLICATES, BLOCK, SEED_CLUSTER, SEED_ITEM = 20000, 500, 12345, 54321
SYSTEMS = ['lexical', 'sapbert', 'sapbert_tag_removed']


def ratio(ne, ee, nn, en):
    if ne == 0 or nn == 0 or ee == 0:
        return float('nan')
    return (en / nn) / (ee / ne)


def interval(values):
    finite = [v for v in values if math.isfinite(v)]
    return [quantile_inverted_cdf(finite, 0.025), quantile_inverted_cdf(finite, 0.975)], len(values) - len(finite)


def evaluate(clusters, items):
    A = [(int(r['exact_items']), int(r['exact_errors']), int(r['narrow_items']), int(r['narrow_errors'])) for r in clusters]
    err_e = [int(r['error']) for r in items if r['label'] == 'EXACT']
    err_n = [int(r['error']) for r in items if r['label'] == 'NARROW']
    tot = [sum(a[j] for a in A) for j in range(4)]
    if tot != [len(err_e), sum(err_e), len(err_n), sum(err_n)]:
        raise ValueError('cluster and item tables disagree')

    g = Generator(SEED_CLUSTER)
    k = len(A)
    cluster_ratios = []
    for _ in range(REPLICATES):
        s = [0, 0, 0, 0]
        for w, a in zip(g.multinomial_uniform(k, k), A):
            if w:
                s[0] += w * a[0]
                s[1] += w * a[1]
                s[2] += w * a[2]
                s[3] += w * a[3]
        cluster_ratios.append(ratio(*s))

    g = Generator(SEED_ITEM)
    nE, nN = len(err_e), len(err_n)
    item_ratios = []
    for start in range(0, REPLICATES, BLOCK):
        b = min(BLOCK, REPLICATES - start)
        ee = [sum(w * e for w, e in zip(g.multinomial_uniform(nE, nE), err_e)) for _ in range(b)]
        en = [sum(w * e for w, e in zip(g.multinomial_uniform(nN, nN), err_n)) for _ in range(b)]
        item_ratios += [ratio(nE, x, nN, y) for x, y in zip(ee, en)]

    (c_lo, c_hi), c_undef = interval(cluster_ratios)
    (i_lo, i_hi), i_undef = interval(item_ratios)
    return {
        'exact_items': tot[0], 'exact_errors': tot[1], 'exact_correct': tot[0] - tot[1], 'exact_error_rate': tot[1] / tot[0],
        'narrow_items': tot[2], 'narrow_errors': tot[3], 'narrow_correct': tot[2] - tot[3], 'narrow_error_rate': tot[3] / tot[2],
        'items': tot[0] + tot[2], 'clusters': k, 'risk_ratio': ratio(*tot), 'error_difference': tot[3] / tot[2] - tot[1] / tot[0],
        'cluster_ci_low': c_lo, 'cluster_ci_high': c_hi, 'cluster_undefined_replicates': c_undef,
        'item_ci_low': i_lo, 'item_ci_high': i_hi, 'item_undefined_replicates': i_undef,
    }


def run(data_dir):
    clusters, items = read_csv(data_dir, 'thai_clusters.csv'), read_csv(data_dir, 'thai_items.csv')
    out = {'replicates': REPLICATES, 'seed_cluster': SEED_CLUSTER, 'seed_item': SEED_ITEM}
    for s in SYSTEMS:
        c = sorted((r for r in clusters if r['system'] == s), key=lambda r: int(r['position']))
        i = sorted((r for r in items if r['system'] == s), key=lambda r: int(r['position']))
        for key, v in evaluate(c, i).items():
            out[s + '_' + key] = v
    return out
