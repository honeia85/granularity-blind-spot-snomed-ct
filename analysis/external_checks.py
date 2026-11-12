"""Cross-checks on separate samples and systems, computed from aggregate tables.

xMEN: SapBERT candidate generation in the xMEN toolkit on separate stratified 800-mention draws for each of the
three Spanish benchmarks, with English translation of the mentions ('translated') or without it ('untranslated');
per benchmark and native label, the number of mentions, of exact matches and the mean Wu-Palmer score.
MedMentions: an 800-mention English sample by depth band of the gold concept (shallow <= 2, mid = 3, deep >= 4),
for the lexical and SapBERT systems of the main analysis and for xMEN; counts of exact matches and of
binary is-a consistency, and the mean Wu-Palmer score.
Participant runs: five public runs of a shared-task participant on MedProcNER, rescored on our graph (domains:
all matched predictions and graph-resolvable predictions) and on the SNOMED CT hierarchy of the reference
release (all matched predictions with numeric identifiers and the resolvable subset); means of exact match and
Wu-Palmer by native label as published with four decimals (or full precision).
Derived quantities: overall means (weighted by the strata sizes), credit gaps (mean Wu-Palmer - mean exact), the
NARROW/EXACT gap ratio, the deep/shallow gap ratio, the NARROW/EXACT exact-match error ratio, and the share of
predictions that are graph-resolvable (resolvable count / all matched count, by label).
"""
import math

from common import BENCHMARKS, read_csv

BANDS = ['shallow', 'mid', 'deep']


def run(data_dir):
    out = {}
    rows = read_csv(data_dir, 'xmen_strata.csv')
    for inp in ('translated', 'untranslated'):
        ratios = []
        for b in BENCHMARKS:
            st = {r['label']: r for r in rows if r['benchmark'] == b and r['input'] == inp}
            n = sum(int(st[k]['n']) for k in st)
            ex = sum(int(st[k]['exact_count']) for k in st)
            wsum = math.fsum(int(st[k]['n']) * float(st[k]['wup_mean']) for k in ('EXACT', 'NARROW'))
            gaps = {k: float(st[k]['wup_mean']) - int(st[k]['exact_count']) / int(st[k]['n']) for k in st}
            pre = 'xmen_%s_%s_' % (inp, b)
            out[pre + 'n'] = n
            out[pre + 'exact'] = ex / n
            out[pre + 'wup'] = wsum / n
            out[pre + 'gap_exact'] = gaps['EXACT']
            out[pre + 'gap_narrow'] = gaps['NARROW']
            out[pre + 'gap_ratio'] = gaps['NARROW'] / gaps['EXACT']
            ratios.append(out[pre + 'gap_ratio'])
        out['xmen_%s_gap_ratio_min' % inp] = min(ratios)
        out['xmen_%s_gap_ratio_max' % inp] = max(ratios)
    rows = read_csv(data_dir, 'medmentions_bands.csv')
    for s in ('lexical', 'sapbert', 'xmen'):
        st = {r['depth_band']: r for r in rows if r['system'] == s}
        n = sum(int(st[k]['n']) for k in BANDS)
        out['medmentions_%s_n' % s] = n
        out['medmentions_%s_exact' % s] = sum(int(st[k]['exact_count']) for k in BANDS) / n
        out['medmentions_%s_is_a' % s] = sum(int(st[k]['is_a_count']) for k in BANDS) / n
        out['medmentions_%s_wup' % s] = math.fsum(int(st[k]['n']) * float(st[k]['wup_mean']) for k in BANDS) / n
        for k in BANDS:
            out['medmentions_%s_gap_%s' % (s, k)] = float(st[k]['wup_mean']) - int(st[k]['exact_count']) / int(st[k]['n'])
            out['medmentions_%s_n_%s' % (s, k)] = int(st[k]['n'])
        out['medmentions_%s_deep_shallow_gap_ratio' % s] = out['medmentions_%s_gap_deep' % s] / out['medmentions_%s_gap_shallow' % s]
        out['medmentions_%s_gap_increases' % s] = int(out['medmentions_%s_gap_shallow' % s] < out['medmentions_%s_gap_mid' % s] < out['medmentions_%s_gap_deep' % s])
    rows = read_csv(data_dir, 'participant_runs.csv')
    runs = sorted({int(r['run']) for r in rows})
    collect = {}
    for k in runs:
        d = {r['domain']: r for r in rows if int(r['run']) == k}
        for dom, r in d.items():
            ge = float(r['wup_mean_exact']) - float(r['exact_mean_exact'])
            gn = float(r['wup_mean_narrow']) - float(r['exact_mean_narrow'])
            pre = 'participant_run%d_%s_' % (k, dom)
            out[pre + 'gap_ratio'] = gn / ge
            out[pre + 'error_ratio'] = (1 - float(r['exact_mean_narrow'])) / (1 - float(r['exact_mean_exact']))
            out[pre + 'n_narrow'] = int(r['n_narrow'])
            collect.setdefault(dom + '_gap_ratio', []).append(gn / ge)
            collect.setdefault(dom + '_error_ratio', []).append(out[pre + 'error_ratio'])
            collect.setdefault(dom + '_n_narrow', []).append(int(r['n_narrow']))
        a, g = d['all_matched'], d['graph_resolvable']
        for lab in ('exact', 'narrow'):
            v = int(g['n_' + lab]) / int(a['n_' + lab])
            out['participant_run%d_resolvable_share_%s' % (k, lab)] = v
            collect.setdefault('resolvable_share_' + lab, []).append(v)
    for name, v in collect.items():
        out['participant_%s_min' % name] = min(v)
        out['participant_%s_max' % name] = max(v)
        out['participant_%s_mean' % name] = math.fsum(v) / len(v)
    out['participant_runs'] = len(runs)
    return out
