"""Translation-by-normalization control on 2,336 mentions, with a document-group bootstrap.

Each mention is scored under four arms, translation (NLLB, Qwen) by normalization (SapBERT retrieval, Qwen
reranking), and under two concept-name generation arms. An arm output is an error (1) when its output could not be
parsed or its code differs from the reference code.

  translation effect under SapBERT    dS = mean(e[nllb_sapbert] - e[qwen_sapbert])
  translation effect under reranking  dQ = mean(e[nllb_rerank] - e[qwen_rerank])
  interaction                         dQ - dS
  generator translation effect        mean(e[nllb_generator] - e[qwen_generator])

Risk ratios divide the NARROW by the EXACT error rate. The Mantel-Haenszel risk ratio of the NLLB-SapBERT arm uses
the benchmarks as strata. Bootstrap: the 246 document groups are resampled with replacement, 10,000 replicates,
random.Random(12345) drawing one group index per position, groups ordered by their number; all mentions of a sampled
group are kept, and the elements at positions floor(0.025 B) and floor(0.975 B) of the sorted replicates form the
interval. The same replicate weights are used for every subset. The sensitivity analysis removes every mention with
an unparsed output in any arm.
"""
import math
import random

from common import BENCHMARKS, mh_ratio_from_tables, read_csv

REPLICATES, SEED = 10000, 12345
RATE_ARMS = ['nllb_sapbert', 'nllb_rerank', 'qwen_sapbert', 'qwen_rerank']
ARMS = RATE_ARMS + ['nllb_generator', 'qwen_generator']
FEATURES = ['n_EXACT', 'n_NARROW'] + [a + s for a in RATE_ARMS for s in ('_err_EXACT', '_err_NARROW')] + ['dS', 'dQ', 'dG']
FI = {f: k for k, f in enumerate(FEATURES)}
NF = len(FEATURES)


def add_features(vec, offset, r):
    lab = r['label']
    e = {a: int(r[a + '_error']) for a in ARMS}
    vec[offset + FI['n_' + lab]] += 1
    for a in RATE_ARMS:
        vec[offset + FI[a + '_err_' + lab]] += e[a]
    vec[offset + FI['dS']] += e['nllb_sapbert'] - e['qwen_sapbert']
    vec[offset + FI['dQ']] += e['nllb_rerank'] - e['qwen_rerank']
    vec[offset + FI['dG']] += e['nllb_generator'] - e['qwen_generator']


def statistics(s):
    n = s[FI['n_EXACT']] + s[FI['n_NARROW']]
    out = {
        'translation_effect_sapbert': s[FI['dS']] / n,
        'translation_effect_rerank': s[FI['dQ']] / n,
        'interaction': (s[FI['dQ']] - s[FI['dS']]) / n,
        'generator_translation_effect': s[FI['dG']] / n,
    }
    ne, nn = s[FI['n_EXACT']], s[FI['n_NARROW']]
    for a in RATE_ARMS:
        ee, en = s[FI[a + '_err_EXACT']], s[FI[a + '_err_NARROW']]
        out[a + '_risk_ratio'] = (en / nn) / (ee / ne) if ee > 0 and ne > 0 and nn > 0 else float('nan')
    return out


def interval(values):
    if not all(math.isfinite(v) for v in values):
        return [None, None]
    s = sorted(values)
    return [s[math.floor(0.025 * len(s))], s[math.floor(0.975 * len(s))]]


def run(data_dir):
    rows = read_csv(data_dir, 'translation_cases.csv')
    numbers = sorted({int(r['document']) for r in rows})
    position = {d: k for k, d in enumerate(numbers)}
    K = len(numbers)
    out = {'mentions': len(rows), 'document_groups': K, 'replicates': REPLICATES, 'seed': SEED}
    for b in BENCHMARKS:
        out['mentions_' + b] = sum(r['benchmark'] == b for r in rows)
    for lab in ('EXACT', 'NARROW'):
        out['mentions_' + lab.lower()] = sum(r['label'] == lab for r in rows)
    for a in ARMS:
        out[a + '_parse_failures'] = sum(int(r[a + '_parse_failure']) for r in rows)
        for lab in ('EXACT', 'NARROW'):
            sub = [r for r in rows if r['label'] == lab]
            out['%s_errors_%s' % (a, lab.lower())] = sum(int(r[a + '_error']) for r in sub)
            out['%s_error_rate_%s' % (a, lab.lower())] = out['%s_errors_%s' % (a, lab.lower())] / len(sub)
        out[a + '_risk_ratio'] = out[a + '_error_rate_narrow'] / out[a + '_error_rate_exact']
    parsed = [not any(int(r[a + '_parse_failure']) for a in ARMS) for r in rows]
    out['mentions_with_any_parse_failure'] = parsed.count(False)
    out['parsed_only_mentions'] = parsed.count(True)

    tables = []
    for b in BENCHMARKS:
        sub = [r for r in rows if r['benchmark'] == b]
        nE = sum(r['label'] == 'EXACT' for r in sub)
        nN = len(sub) - nE
        eE = sum(int(r['nllb_sapbert_error']) for r in sub if r['label'] == 'EXACT')
        eN = sum(int(r['nllb_sapbert_error']) for r in sub if r['label'] == 'NARROW')
        tables.append((nE, eE, nN, eN))
        key = 'nllb_sapbert_' + b
        out[key + '_exact_mentions'], out[key + '_exact_errors'], out[key + '_narrow_mentions'], out[key + '_narrow_errors'] = nE, eE, nN, eN
        out[key + '_risk_ratio'] = (eN / nN) / (eE / nE)
    out['nllb_sapbert_mh_risk_ratio'] = mh_ratio_from_tables(tables)[0]

    # subsets: all mentions, each benchmark, mentions without unparsed output; one feature block per subset
    subsets = [('', lambda r, ok: True)] + [(b + '_', (lambda b: lambda r, ok: r['benchmark'] == b)(b)) for b in BENCHMARKS] \
        + [('parsed_only_', lambda r, ok: ok)]
    per_doc = [[0] * (NF * len(subsets)) for _ in range(K)]
    for r, ok in zip(rows, parsed):
        for j, (_, keep) in enumerate(subsets):
            if keep(r, ok):
                add_features(per_doc[position[int(r['document'])]], j * NF, r)
    total = [sum(v[c] for v in per_doc) for c in range(NF * len(subsets))]

    rng = random.Random(SEED)
    draws = {j: [] for j in range(len(subsets))}
    for _ in range(REPLICATES):
        w = [0] * K
        for _ in range(K):
            w[rng.randrange(K)] += 1
        acc = [0] * (NF * len(subsets))
        for k in range(K):
            if w[k]:
                wk, v = w[k], per_doc[k]
                acc = [x + wk * y for x, y in zip(acc, v)]
        for j in range(len(subsets)):
            draws[j].append(statistics(acc[j * NF:(j + 1) * NF]))

    for j, (prefix, _) in enumerate(subsets):
        point = statistics(total[j * NF:(j + 1) * NF])
        for name, value in point.items():
            out[prefix + name] = value
            out[prefix + name + '_ci_low'], out[prefix + name + '_ci_high'] = interval([d[name] for d in draws[j]])
    return out
