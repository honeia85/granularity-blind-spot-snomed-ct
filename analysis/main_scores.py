"""Main analysis of the fixed panel: scores, credit gaps, error incidence, decomposition and relation types.

Data: the 2,103 fixed-panel mentions (678 DisTEMIST, 625 MedProcNER, 800 SympTEMIST) with their native-corpus
granularity label (EXACT or NARROW), and for every mention the outcome of each system in both candidate spaces
(oracle: the gold-inclusive inventory; train: the training-only inventory): exact match (0/1), binary
is-a consistency (0/1), Wu-Palmer score and the ontology relation of the prediction to the gold concept.
Five systems (lexical TF-IDF, SapBERT, fine-tuned SapBERT, LLM reranker, LLM generator) give 14 cells per space;
fine-tuned SapBERT was not run on MedProcNER.

Definitions, within a cell and stratum (EXACT or NARROW mentions):
  credit gap      = mean(Wu-Palmer) - mean(exact); ratio = gap(NARROW) / gap(EXACT); difference = gap(N) - gap(E)
  error rate      = share of mentions with exact = 0; incidence ratio = rate(NARROW) / rate(EXACT)
  conditional WuP = mean Wu-Palmer among errors; proximity ratio = conditional WuP(NARROW) / conditional WuP(EXACT)
  uplift          = mean(is-a) - mean(exact)
Because the gap equals the error rate times the conditional Wu-Palmer, the gap ratio is the product of the
incidence and proximity ratios. Geometric means give equal weight to the 14 cells of a space.
Pooled measures: Mantel-Haenszel risk ratio and odds ratio of error, NARROW versus EXACT, over the cells as strata;
per system, over the benchmarks on which it was run. Breslow-Day homogeneity test within system across
benchmarks (chi-square upper tail; Holm adjustment over the five systems). Relation types: each non-exact
prediction is sibling/cousin, descendant (prediction more specific than gold), ancestor or unresolved; shares
of counts and of summed Wu-Palmer credit. Fine-tuning: paired gains of fine-tuned over off-the-shelf SapBERT and
the exact two-sided McNemar test on discordant mentions.
"""
import math

from common import BENCHMARKS, read_csv

SYSTEMS = ['lexical', 'sapbert', 'sapbert_ft', 'llm_rerank', 'llm_gen']
SPACES = ['oracle', 'train']
RELATIONS = ['sibling', 'descendant', 'ancestor', 'unresolved']
WUP_SCALE = 10000  # Wu-Palmer scores are given with at most four decimals


def wup_units(text):
    """Wu-Palmer score as an integer number of 1/10000 units (exact)."""
    whole, _, frac = text.partition('.')
    frac = (frac + '0000')[:4]
    assert len(text.partition('.')[2]) <= 4
    return int(whole) * WUP_SCALE + int(frac)


def load_main(data_dir):
    """Returns (items, cells). items: per benchmark, list of item dicts in position order.

    cells: {(space, benchmark, system): list of rows in the same order}; a row is
    (narrow 0/1, exact 0/1, is_a 0/1, Wu-Palmer units, relation type).
    """
    items = read_csv(data_dir, 'panel_items.csv')
    by_bench = {b: sorted((r for r in items if r['benchmark'] == b), key=lambda r: int(r['position'])) for b in BENCHMARKS}
    where = {}
    for b in BENCHMARKS:
        for k, r in enumerate(by_bench[b]):
            assert int(r['position']) == k + 1
            where[r['item_id']] = (b, k)
    cells = {}
    for r in read_csv(data_dir, 'mention_scores.csv'):
        b, k = where[r['item_id']]
        key = (r['candidate_space'], b, r['system'])
        if key not in cells:
            cells[key] = [None] * len(by_bench[b])
        narrow = 1 if by_bench[b][k]['label'] == 'NARROW' else 0
        cells[key][k] = (narrow, int(r['exact']), int(r['is_a']), wup_units(r['wup']), r['relation_type'])
    for key, rows in cells.items():
        assert all(x is not None for x in rows), key
    return by_bench, cells


def cell_keys(space):
    return [(space, b, s) for b in BENCHMARKS for s in SYSTEMS if (space, b, s) in CELLS_PRESENT]


CELLS_PRESENT = set()


def stratum_sums(rows, narrow):
    """Sums over one stratum: n, errors, exact, is_a, Wu-Palmer units, Wu-Palmer units among errors."""
    n = e = ex = isa = w = we = 0
    for r in rows:
        if r[0] != narrow:
            continue
        n += 1
        ex += r[1]
        isa += r[2]
        w += r[3]
        if r[1] == 0:
            e += 1
            we += r[3]
    return {'n': n, 'errors': e, 'exact': ex, 'is_a': isa, 'wup': w, 'wup_errors': we}


def gap(s):
    """Credit gap from stratum sums: mean Wu-Palmer minus mean exact (equals Wu-Palmer units among errors / n)."""
    return (s['wup'] - s['exact'] * WUP_SCALE) / WUP_SCALE / s['n']


def cell_measures(rows):
    E, N = stratum_sums(rows, 0), stratum_sums(rows, 1)
    n = E['n'] + N['n']
    out = {
        'n': n, 'exact_items': E['n'], 'narrow_items': N['n'],
        'exact': (E['exact'] + N['exact']) / n,
        'is_a': (E['is_a'] + N['is_a']) / n,
        'wup': (E['wup'] + N['wup']) / WUP_SCALE / n,
        'gap_exact': gap(E), 'gap_narrow': gap(N),
        'errors_exact': E['errors'], 'errors_narrow': N['errors'],
        'error_rate_exact': E['errors'] / E['n'], 'error_rate_narrow': N['errors'] / N['n'],
        'conditional_wup_exact': E['wup_errors'] / WUP_SCALE / E['errors'],
        'conditional_wup_narrow': N['wup_errors'] / WUP_SCALE / N['errors'],
        'exact_exact': E['exact'] / E['n'], 'exact_narrow': N['exact'] / N['n'],
        'is_a_exact': E['is_a'] / E['n'], 'is_a_narrow': N['is_a'] / N['n'],
        'uplift_exact': (E['is_a'] - E['exact']) / E['n'], 'uplift_narrow': (N['is_a'] - N['exact']) / N['n'],
    }
    out['wup_minus_exact'] = ((E['wup'] + N['wup']) - (E['exact'] + N['exact']) * WUP_SCALE) / WUP_SCALE / n
    out['gap_ratio'] = out['gap_narrow'] / out['gap_exact']
    out['gap_difference'] = out['gap_narrow'] - out['gap_exact']
    out['incidence_ratio'] = out['error_rate_narrow'] / out['error_rate_exact']
    a, c = N['errors'], N['n'] - N['errors']
    b, d = E['errors'], E['n'] - E['errors']
    out['incidence_odds_ratio'] = (a * d) / (b * c)
    out['proximity_ratio'] = out['conditional_wup_narrow'] / out['conditional_wup_exact']
    out['uplift_difference'] = out['uplift_narrow'] - out['uplift_exact']
    out['table'] = (a, b, c, d)
    return out


def mh_risk_ratio(tables):
    """tables: (errors NARROW, errors EXACT, correct NARROW, correct EXACT)."""
    num = den = 0.0
    for a, b, c, d in tables:
        n1, n0 = a + c, b + d
        if n1 and n0:
            num += a * n0 / (n1 + n0)
            den += b * n1 / (n1 + n0)
    return num / den


def mh_odds_ratio(tables):
    num = den = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n:
            num += a * d / n
            den += b * c / n
    return num / den


def chi2_upper(x, df):
    """Upper tail of the chi-square distribution (regularized incomplete gamma: series or continued fraction)."""
    if x <= 0:
        return 1.0
    a, h = df / 2.0, x / 2.0
    if h < a + 1.0:
        term = total = 1.0 / a
        k = a
        for _ in range(500):
            k += 1.0
            term *= h / k
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return 1.0 - total * math.exp(-h + a * math.log(h) - math.lgamma(a))
    b = h + 1.0 - a
    c, d = 1e300, 1.0 / b
    f = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        step = d * c
        f *= step
        if abs(step - 1.0) < 1e-14:
            break
    return f * math.exp(-h + a * math.log(h) - math.lgamma(a))


def breslow_day(tables, common_or):
    """Breslow-Day statistic, its degrees of freedom and upper-tail P value."""
    stat, used = 0.0, 0
    for a, b, c, d in tables:
        n1, n2 = a + c, b + d
        m1 = a + b
        n = n1 + n2
        if min(n1, n2, m1, c + d) == 0:
            continue
        A = common_or - 1.0
        Bq = -(common_or * (n1 + m1) + (n2 - m1))
        C = common_or * n1 * m1
        if abs(A) < 1e-12:
            fit = n1 * m1 / n
        else:
            root = math.sqrt(max(Bq * Bq - 4 * A * C, 0.0))
            fit = (-Bq - root) / (2 * A)
            if not (max(0.0, m1 - n2) < fit < min(n1, m1)):
                fit = (-Bq + root) / (2 * A)
        cells = (fit, m1 - fit, n1 - fit, n2 - m1 + fit)
        if min(cells) <= 0:
            continue
        stat += (a - fit) ** 2 * (1.0 / cells[0] + 1.0 / cells[1] + 1.0 / cells[2] + 1.0 / cells[3])
        used += 1
    df = used - 1
    return stat, df, chi2_upper(stat, df)


def mcnemar_exact(b, c):
    """Exact two-sided McNemar P value: twice the binomial(b + c, 1/2) tail of the smaller count, at most 1."""
    n, k = b + c, min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def geometric_mean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def relation_profile(rows_list, narrow=None):
    """Counts and Wu-Palmer units by relation type among non-exact predictions of the given rows."""
    count = {r: 0 for r in RELATIONS}
    credit = {r: 0 for r in RELATIONS}
    for rows in rows_list:
        for x in rows:
            if x[1] == 1 or (narrow is not None and x[0] != narrow):
                continue
            assert x[4] in count
            count[x[4]] += 1
            credit[x[4]] += x[3]
    return count, credit


def run(data_dir):
    by_bench, cells = load_main(data_dir)
    CELLS_PRESENT.clear()
    CELLS_PRESENT.update(cells)
    out = {}
    # sample
    docs_all = set()
    for b in BENCHMARKS:
        its = by_bench[b]
        nn = sum(1 for r in its if r['label'] == 'NARROW')
        out['items_' + b] = len(its)
        out['narrow_items_' + b] = nn
        out['exact_items_' + b] = len(its) - nn
        out['narrow_share_' + b] = nn / len(its)
        docs = {r['document_id'] for r in its}
        out['documents_' + b] = len(docs)
        docs_all |= docs
    out['items'] = sum(out['items_' + b] for b in BENCHMARKS)
    out['narrow_items'] = sum(out['narrow_items_' + b] for b in BENCHMARKS)
    out['exact_items'] = out['items'] - out['narrow_items']
    out['narrow_share'] = out['narrow_items'] / out['items']
    out['documents'] = len(docs_all)
    out['cells_oracle'] = len(cell_keys('oracle'))
    out['cells_all'] = len(cells)
    for b in BENCHMARKS:
        out['oracle_rows_' + b] = sum(len(v) for k, v in cells.items() if k[0] == 'oracle' and k[1] == b)
    out['score_rows'] = sum(len(v) for v in cells.values())

    measures = {}
    for space in SPACES:
        keys = cell_keys(space)
        for key in keys:
            m = cell_measures(cells[key])
            measures[key] = m
            prefix = '%s_%s_%s_' % key
            for name, value in m.items():
                if name != 'table':
                    out[prefix + name] = value
        tables = [measures[k]['table'] for k in keys]
        out[space + '_mh_risk_ratio'] = mh_risk_ratio(tables)
        out[space + '_mh_odds_ratio'] = mh_odds_ratio(tables)
        A = sum(t[0] for t in tables)
        B = sum(t[1] for t in tables)
        C = sum(t[2] for t in tables)
        D = sum(t[3] for t in tables)
        out[space + '_crude_odds_ratio'] = (A * D) / (B * C)
        for name, field in (('gap_ratio', 'gap_ratio'), ('incidence_ratio', 'incidence_ratio'), ('proximity_ratio', 'proximity_ratio')):
            vals = [measures[k][field] for k in keys]
            out['%s_geometric_mean_%s' % (space, name)] = geometric_mean(vals)
            out['%s_%s_min' % (space, name)] = min(vals)
            out['%s_%s_max' % (space, name)] = max(vals)
            out['%s_cells_%s_above_1' % (space, name)] = sum(1 for v in vals if v > 1)
            out['%s_cells_%s_at_least_1' % (space, name)] = sum(1 for v in vals if v >= 1)
        ors = [measures[k]['incidence_odds_ratio'] for k in keys]
        out[space + '_cell_odds_ratio_min'] = min(ors)
        out[space + '_cell_odds_ratio_max'] = max(ors)
        out[space + '_cells_narrow_gap_larger'] = sum(1 for k in keys if measures[k]['gap_narrow'] > measures[k]['gap_exact'])
        out[space + '_cells_is_a_lower_on_narrow'] = sum(1 for k in keys if measures[k]['is_a_narrow'] < measures[k]['is_a_exact'])
        out[space + '_cells_uplift_difference_positive'] = sum(1 for k in keys if measures[k]['uplift_difference'] > 0)
        out[space + '_cells_conditional_wup_difference_positive'] = sum(
            1 for k in keys if measures[k]['conditional_wup_narrow'] > measures[k]['conditional_wup_exact'])
        diffs = [measures[k]['conditional_wup_narrow'] - measures[k]['conditional_wup_exact'] for k in keys]
        out[space + '_conditional_wup_difference_mean'] = math.fsum(diffs) / len(diffs)
        for b in BENCHMARKS:
            d = [measures[k]['conditional_wup_narrow'] - measures[k]['conditional_wup_exact'] for k in keys if k[1] == b]
            out['%s_conditional_wup_difference_mean_%s' % (space, b)] = math.fsum(d) / len(d)
            out['%s_cells_%s' % (space, b)] = len(d)
        # per system across benchmarks
        holm = []
        for s in SYSTEMS:
            ks = [k for k in keys if k[2] == s]
            t = [measures[k]['table'] for k in ks]
            out['%s_%s_mh_risk_ratio' % (space, s)] = mh_risk_ratio(t)
            orc = mh_odds_ratio(t)
            out['%s_%s_mh_odds_ratio' % (space, s)] = orc
            stat, df, p = breslow_day(t, orc)
            out['%s_%s_breslow_day_chi2' % (space, s)] = stat
            out['%s_%s_breslow_day_df' % (space, s)] = df
            out['%s_%s_breslow_day_p' % (space, s)] = p
            holm.append((p, s))
        running = 0.0
        for rank, (p, s) in enumerate(sorted(holm)):
            running = max(running, min(1.0, p * (len(holm) - rank)))
            out['%s_%s_breslow_day_p_holm' % (space, s)] = running
        out[space + '_breslow_day_p_below_005'] = sum(1 for p, _ in holm if p < 0.05)
        out[space + '_breslow_day_p_holm_below_005'] = sum(1 for s in SYSTEMS if out['%s_%s_breslow_day_p_holm' % (space, s)] < 0.05)
        rrs = [out['%s_%s_mh_risk_ratio' % (space, s)] for s in SYSTEMS]
        out[space + '_system_mh_risk_ratio_min'] = min(rrs)
        out[space + '_system_mh_risk_ratio_max'] = max(rrs)
        for s in SYSTEMS:
            v = [measures[k]['gap_ratio'] for k in keys if k[2] == s]
            out['%s_%s_gap_ratio_min' % (space, s)] = min(v)
            out['%s_%s_gap_ratio_max' % (space, s)] = max(v)
        strong = [measures[k]['gap_ratio'] for k in keys if k[2] in ('sapbert', 'sapbert_ft', 'llm_rerank')]
        out[space + '_strong_systems_gap_ratio_min'] = min(strong)
        out[space + '_strong_systems_gap_ratio_max'] = max(strong)

    # relation types, gold-inclusive space
    okeys = cell_keys('oracle')
    sib_count, sib_credit = [], []
    for s in SYSTEMS:
        rows_list = [cells[k] for k in okeys if k[2] == s]
        count, credit = relation_profile(rows_list)
        total, total_credit = sum(count.values()), sum(credit.values())
        out['relations_%s_non_exact' % s] = total
        for r in RELATIONS:
            out['relations_%s_%s' % (s, r)] = count[r]
            out['relations_%s_%s_share' % (s, r)] = count[r] / total
            out['relations_%s_%s_credit_share' % (s, r)] = credit[r] / total_credit
        sib_count.append(count['sibling'] / total)
        sib_credit.append(credit['sibling'] / total_credit)
    out['relations_sibling_share_min'] = min(sib_count)
    out['relations_sibling_share_max'] = max(sib_count)
    out['relations_sibling_credit_share_min'] = min(sib_credit)
    out['relations_sibling_credit_share_max'] = max(sib_credit)
    # pooled LLM reranker and generator profile, by benchmark and stratum
    qwen_total = 0
    shares = {'sibling': [], 'descendant': [], 'ancestor': []}
    for b in BENCHMARKS:
        rows_list = [cells[k] for k in okeys if k[1] == b and k[2] in ('llm_rerank', 'llm_gen')]
        count, credit = relation_profile(rows_list)
        total, total_credit = sum(count.values()), sum(credit.values())
        qwen_total += total
        pre = 'llm_pooled_%s_' % b
        out[pre + 'non_exact'] = total
        out[pre + 'sibling_share'] = count['sibling'] / total
        out[pre + 'sibling_mean_wup'] = credit['sibling'] / WUP_SCALE / count['sibling']
        out[pre + 'sibling_credit_share'] = credit['sibling'] / total_credit
        out[pre + 'ancestor_descendant_credit_share'] = (credit['ancestor'] + credit['descendant']) / total_credit
        per = {}
        for label, narrow in (('exact', 0), ('narrow', 1)):
            c2, w2 = relation_profile(rows_list, narrow)
            t2, tw2 = sum(c2.values()), sum(w2.values())
            for r in RELATIONS:
                out[pre + '%s_%s_share' % (label, r)] = c2[r] / t2
                if r in shares:
                    shares[r].append(c2[r] / t2)
            out[pre + '%s_sibling_credit_share' % label] = w2['sibling'] / tw2
            per[label] = c2['descendant'] / t2
        out[pre + 'descendant_share_difference'] = per['narrow'] - per['exact']
    out['llm_pooled_non_exact'] = qwen_total
    for r, v in shares.items():
        out['llm_pooled_%s_share_min' % r] = min(v)
        out['llm_pooled_%s_share_max' % r] = max(v)
    ex_desc = [out['llm_pooled_%s_exact_descendant_share' % b] for b in BENCHMARKS]
    out['llm_pooled_exact_descendant_share_min'] = min(ex_desc)
    out['llm_pooled_exact_descendant_share_max'] = max(ex_desc)
    # sibling share of NARROW non-exact credit, per cell
    enc = []
    for k in okeys:
        count, credit = relation_profile([cells[k]], 1)
        v = credit['sibling'] / sum(credit.values())
        out['%s_%s_%s_narrow_sibling_credit_share' % k] = v
        if k[2] in ('sapbert', 'sapbert_ft') and k[1] != 'DisTEMIST':
            enc.append(v)
    # LLM reranker and generator cells, by stratum
    per_cell = {'sibling': [], 'descendant': [], 'ancestor': []}
    exact_desc = []
    for k in okeys:
        if k[2] not in ('llm_rerank', 'llm_gen'):
            continue
        for narrow in (0, 1):
            count, _ = relation_profile([cells[k]], narrow)
            t = sum(count.values())
            for r in per_cell:
                per_cell[r].append(count[r] / t)
            if narrow == 0:
                exact_desc.append(count['descendant'] / t)
    for r, v in per_cell.items():
        out['llm_cells_%s_share_min' % r] = min(v)
        out['llm_cells_%s_share_max' % r] = max(v)
    out['llm_cells_exact_descendant_share_min'] = min(exact_desc)
    out['llm_cells_exact_descendant_share_max'] = max(exact_desc)
    out['encoder_narrow_sibling_credit_share_min_other_benchmarks'] = min(enc)
    out['encoder_narrow_sibling_credit_share_max_other_benchmarks'] = max(enc)

    # fine-tuning, gold-inclusive space
    for b in ('DisTEMIST', 'SympTEMIST'):
        base, ft = cells[('oracle', b, 'sapbert')], cells[('oracle', b, 'sapbert_ft')]
        gain = sum(1 for x, y in zip(base, ft) if x[1] == 0 and y[1] == 1)
        loss = sum(1 for x, y in zip(base, ft) if x[1] == 1 and y[1] == 0)
        out['finetune_%s_discordant_gain' % b] = gain
        out['finetune_%s_discordant_loss' % b] = loss
        out['finetune_%s_mcnemar_p' % b] = mcnemar_exact(gain, loss)
        for label, narrow in (('exact', 0), ('narrow', 1)):
            n = sum(1 for x in base if x[0] == narrow)
            d = sum(y[1] - x[1] for x, y in zip(base, ft) if x[0] == narrow)
            out['finetune_%s_%s_gain_points' % (b, label)] = 100.0 * d / n
    return out
