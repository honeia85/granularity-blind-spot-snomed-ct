"""Spanish lexical retrieval without translation: exact matches with a public gazetteer and with Spanish-edition synonyms.

Counts by benchmark and label are computed from the mention-level outcomes and compared with the count table
written by licensed_inputs/spanish_lexical_retrieval.py (whose candidate-code counts need the licensed inputs).
The error risk ratio is the NARROW error rate divided by the EXACT error rate, an error being a mention without an
exact match.
"""
from common import BENCHMARKS, read_csv

COLUMNS = ['n_EXACT', 'n_NARROW', 'gazetteer_exact_EXACT', 'gazetteer_exact_NARROW', 'spanish_edition_exact_EXACT', 'spanish_edition_exact_NARROW']


def run(data_dir):
    mentions = read_csv(data_dir, 'spanish_lexical_mentions.csv')
    table = {r['benchmark']: r for r in read_csv(data_dir, 'spanish_lexical_counts.csv')}
    out = {'mentions': len(mentions)}
    ratios = []
    for b in BENCHMARKS:
        sub = [m for m in mentions if m['benchmark'] == b]
        c = {}
        for lab in ('EXACT', 'NARROW'):
            g = [m for m in sub if m['label'] == lab]
            c['n_' + lab] = len(g)
            c['gazetteer_exact_' + lab] = sum(int(m['gazetteer_exact']) for m in g)
            c['spanish_edition_exact_' + lab] = sum(int(m['spanish_edition_exact']) for m in g)
        if any(int(table[b][k]) != c[k] for k in COLUMNS):
            raise ValueError('mention outcomes disagree with the count table for ' + b)
        out['candidate_codes_' + b] = int(table[b]['candidate_codes'])
        for k in COLUMNS:
            out[k + '_' + b] = c[k]
        for res in ('gazetteer', 'spanish_edition'):
            rE = 1 - c[res + '_exact_EXACT'] / c['n_EXACT']
            rN = 1 - c[res + '_exact_NARROW'] / c['n_NARROW']
            out['%s_risk_ratio_%s' % (res, b)] = rN / rE
            ratios.append(rN / rE)
    out['risk_ratio_min'] = min(ratios)
    out['risk_ratio_max'] = max(ratios)
    return out
