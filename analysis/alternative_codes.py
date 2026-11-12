"""Alternative acceptable codes for NARROW mentions: agreement of two raters (A and B), adjudication and a design-weighted estimate.

Two raters (A and B) decided independently, for each of 100 NARROW mentions (a simple random sample, stratified by
benchmark: 41 DisTEMIST, 24 MedProcNER, 35 SympTEMIST), whether at least one presented candidate concept is an
acceptable alternative to the assigned code (decision 'selected', 'none' or 'hold'; selected candidates are listed
per mention). Agreement is computed on the
binary decision (selected or not) and on the three categories; kappa = (po - pe) / (1 - pe). Two judgments
disagree when the decisions differ or when both select candidates but the sets of candidates differ. The final
judgment is the agreed judgment, or the consensus judgment for a disagreement. The sampling frame of each benchmark
is the set of NARROW mentions of the fixed panel. The design-weighted proportion is sum_h W_h p_h with
W_h = N_h / N, its variance sum_h W_h^2 (1 - n_h / N_h) p_h (1 - p_h) / (n_h - 1), and the 95% interval is
computed on the logit scale.
"""
import math

from common import BENCHMARKS, Z95, kappa, read_csv


def run(data_dir):
    rows = read_csv(data_dir, 'alternative_code_judgments.csv')
    panel = read_csv(data_dir, 'panel_items.csv')
    frame = {b: sum(1 for r in panel if r['benchmark'] == b and r['label'] == 'NARROW') for b in BENCHMARKS}
    cset = lambda s: frozenset(t for t in s.split(';') if t)

    out = {}
    n = len(rows)
    sel1 = [r['rater_A_decision'] == 'selected' for r in rows]
    sel2 = [r['rater_B_decision'] == 'selected' for r in rows]
    out['items'] = n
    out['rater_A_selected'] = sum(sel1)
    out['rater_B_selected'] = sum(sel2)
    for b in BENCHMARKS:
        out['rater_A_selected_' + b] = sum(1 for r, s in zip(rows, sel1) if s and r['benchmark'] == b)
        out['rater_B_selected_' + b] = sum(1 for r, s in zip(rows, sel2) if s and r['benchmark'] == b)
    out['selected_and_held'] = sum(1 for r in rows if {r['rater_A_decision'], r['rater_B_decision']} == {'selected', 'hold'})
    binary = [(a, b) for a, b in zip(sel1, sel2)]
    out['binary_agreements'] = sum(a == b for a, b in binary)
    three = [(r['rater_A_decision'], r['rater_B_decision']) for r in rows]
    out['three_category_agreements'] = sum(a == b for a, b in three)
    out['kappa_binary'] = kappa(binary, [True, False])['kappa']
    out['kappa_three_category'] = kappa(three, ['selected', 'none', 'hold'])['kappa']

    both = [r for r, a, b in zip(rows, sel1, sel2) if a and b]
    out['both_selected'] = len(both)
    out['both_selected_same_candidates'] = sum(1 for r in both if cset(r['rater_A_candidates']) == cset(r['rater_B_candidates']))
    differ_decision = [r for r in rows if r['rater_A_decision'] != r['rater_B_decision']]
    differ_set = [r for r in both if cset(r['rater_A_candidates']) != cset(r['rater_B_candidates'])]
    disagreements = {r['item_id'] for r in differ_decision} | {r['item_id'] for r in differ_set}
    out['disagreements'] = len(disagreements)
    out['disagreements_decision'] = len(differ_decision)
    out['disagreements_candidates'] = len(differ_set)
    with_consensus = {r['item_id'] for r in rows if r['consensus_decision']}
    if with_consensus != disagreements:
        raise ValueError('consensus judgments must cover exactly the disagreements')

    final = {}
    for r in rows:
        decision = r['consensus_decision'] if r['item_id'] in disagreements else r['rater_A_decision']
        final[r['item_id']] = 1 if decision == 'selected' else 0
    out['final_selected'] = sum(final.values())
    out['final_selected_share'] = out['final_selected'] / n

    total = sum(frame.values())
    out['frame'] = total
    p = var = 0.0
    for b in BENCHMARKS:
        sub = [r for r in rows if r['benchmark'] == b]
        nh, Nh = len(sub), frame[b]
        sh = sum(final[r['item_id']] for r in sub)
        ph = sh / nh
        out['frame_' + b] = Nh
        out['sampled_' + b] = nh
        out['final_selected_' + b] = sh
        out['proportion_' + b] = ph
        out['inclusion_probability_' + b] = nh / Nh
        W = Nh / total
        p += W * ph
        var += W * W * (1 - nh / Nh) * ph * (1 - ph) / (nh - 1)
    se = math.sqrt(var)
    logit, se_logit = math.log(p / (1 - p)), se / (p * (1 - p))
    out['weighted_proportion'] = p
    out['weighted_proportion_se'] = se
    out['weighted_proportion_ci_low'] = 1 / (1 + math.exp(-(logit - Z95 * se_logit)))
    out['weighted_proportion_ci_high'] = 1 / (1 + math.exp(-(logit + Z95 * se_logit)))
    return out
