"""Terminology release: summaries of the aggregate tables in data/terminology_release.

The tables were produced from licensed SNOMED CT release files and the fixed predictions by
licensed_inputs/terminology_release_tables.py. This module checks their internal consistency (the geometric mean of
the cell-level credit-gap ratios, the row-change totals and the number of cells with changed rows) and reports the
quantities cited in the article. Release dates are kept as YYYY-MM-DD strings. The credit-gap ratio of a cell is the mean partial-credit gap (Wu-Palmer score
minus exact match) of its NARROW rows divided by that of its EXACT rows.
"""
import math

from common import BENCHMARKS, plain_sum, read_csv


def run(data_dir):
    d = lambda name: read_csv(data_dir, 'terminology_release/' + name)
    points = d('time_points.csv')
    cells = d('cells.csv')
    summary = {r['quantity']: r['value'] for r in d('summary.csv')}
    out = {}
    for key in ('reference_release', 'spanish_release', 'spanish_declared_international_base'):
        out[key] = summary[key]
    for key in ('prediction_rows', 'evaluated_concepts', 'gold_concepts', 'predicted_concepts', 'spanish_extension_concepts', 'spanish_extension_is_a_rows',
                'spanish_extension_active_is_a_rows', 'spanish_extension_inactive_is_a_rows', 'spanish_extension_concepts_among_evaluated',
                'spanish_extension_rows_changed_when_added_to_base'):
        out[key] = int(summary[key])
    international = [p for p in points if p['time_point'].startswith('international_')]
    out['international_time_points'] = len(international)
    out['first_international_release'] = min(p['international_release'] for p in international)
    out['last_international_release'] = max(p['international_release'] for p in international)
    gms = []
    for p in points:
        t = p['time_point']
        key = 'spanish_edition' if t == 'spanish_edition' else t.split('_')[1]
        for col in ('inferred_edges', 'inferred_edges_added_vs_reference', 'inferred_edges_deleted_vs_reference', 'stated_edges',
                    'stated_edges_added_vs_reference', 'stated_edges_deleted_vs_reference', 'gold_active', 'gold_inactive', 'gold_absent',
                    'rows_changed_vs_reference', 'rows_up', 'rows_down', 'cells_with_changed_rows'):
            out['%s_%s' % (key, col)] = int(p[col])
        gm = float(p['credit_gap_ratio_geometric_mean'])
        out[key + '_credit_gap_ratio'] = gm
        if t != 'spanish_edition':
            out['time_point_' + key] = p['international_release']
        if t == 'spanish_edition':
            out['spanish_edition_international_base'] = p['international_release']
            continue
        sub = [c for c in cells if c['time_point'] == t]
        ratios = [(float(c['credit_gap_narrow'])) / float(c['credit_gap_exact']) for c in sub]
        recomputed = math.exp(plain_sum(math.log(x) for x in ratios) / len(ratios))
        if abs(recomputed - gm) > 1e-12:
            raise ValueError('geometric mean of cell ratios differs from the time-point table at ' + t)
        if sum(int(c['rows_changed_vs_reference']) for c in sub) != int(p['rows_changed_vs_reference']):
            raise ValueError('row changes do not add up at ' + t)
        if sum(1 for c in sub if int(c['rows_changed_vs_reference']) > 0) != int(p['cells_with_changed_rows']):
            raise ValueError('cells with changed rows do not match at ' + t)
        if sum(int(c['exact_rows']) + int(c['narrow_rows']) for c in sub) != out['prediction_rows']:
            raise ValueError('cells do not cover all prediction rows at ' + t)
        out[key + '_cells'] = len(sub)
        out[key + '_cell_ratio_min'] = min(ratios)
        gms.append(gm)
    out['credit_gap_ratio_min_over_international'] = min(gms)
    out['credit_gap_ratio_max_over_international'] = max(gms)
    for r in d('gold_states_by_benchmark.csv'):
        key = r['time_point'].split('_')[1] + '_' + r['benchmark']
        for col in ('gold_concepts', 'active', 'inactive', 'absent'):
            out['%s_gold_%s' % (key, col.replace('gold_', ''))] = int(r[col])
    for r in d('compatible_releases.csv'):
        b = r['benchmark']
        out['window_%s_gold_concepts' % b] = int(r['gold_concepts'])
        out['window_%s_releases_all_active' % b] = int(r['releases_all_exist_and_active'])
        out['window_%s_gold_inactive_at_20221031' % b] = int(r['gold_inactive_at_20221031'])
        for col in ('first_release_all_exist', 'first_release_all_active', 'last_release_all_active'):
            out['window_%s_%s' % (b, col)] = r[col] or None
    for r in d('edition_statements.csv'):
        if r['date']:
            out['statement_%s' % r['benchmark']] = r['date']
    return out
