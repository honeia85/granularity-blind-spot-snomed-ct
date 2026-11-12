#!/usr/bin/env python3
"""Hierarchical relation of the accepted alternative codes and their effect on the NARROW exact-match penalty.

Usage:
  python3 alternative_code_hierarchy.py --international INTERNATIONAL_FULL_RELEASE.zip \
      --review alternative_code_review.csv --isa-type-id ID --inferred-characteristic-id ID \
      [--data ../data] --out results.json

Inputs (none is distributed with the package):
  --international  SNOMED CT International Edition release archive (RF2) of 1 November 2026 or later, whose Full
                   files cover both time points used here (17 October 2025 and 1 November 2026)
  --review         CSV, one row per reviewed NARROW mention, in the sampling order within each benchmark, with the
                   columns benchmark, gold_code, candidate_codes (all presented candidates), accepted_codes (codes
                   accepted as alternatives in the final judgment: the agreed judgment of raters A and B, or the
                   consensus judgment), and predicted_lexical, predicted_sapbert, predicted_sapbert_ft,
                   predicted_llm_rerank, predicted_llm_gen (top-ranked prediction of each system in the gold-inclusive
                   candidate space; empty where the system was not evaluated on the benchmark). Lists are separated
                   by ';'.
  --isa-type-id, --inferred-characteristic-id
                   identifiers of two SNOMED CT metadata concepts: the is-a relationship type and the inferred
                   characteristic type. They are not written in this file; take them from the documentation of the
                   SNOMED CT release you use.
  --data           the data folder of this package (default: ../data next to this file); panel_items.csv and
                   mention_scores.csv give the NARROW and EXACT error counts of the 14 gold-inclusive cells, and
                   alternative_code_judgments.csv the numbers of mentions with an accepted alternative.
Output: one JSON file of aggregate counts, proportions and risk ratios (no identifier and no code).

Method.
Hierarchy. For a time point T, every relationship and concept takes its latest version with effectiveTime <= T; the
hierarchy consists of the active inferred is-a relationships, and the root is the only concept that is a parent but
has no parent. Depth is the longest is-a path from the root. A concept is usable at T if it is active and has a
parent. For a reference (gold) concept g and an alternative c, both usable, c is an ancestor of g if it lies on an
is-a path above g (hops = shortest number of is-a steps), a descendant if g lies above c, a sibling if the two share
a direct parent, and otherwise lateral when their deepest common ancestor is not the root or in another top-level
hierarchy when the root is their only common ancestor. For siblings and lateral pairs the script also records the
depth of the deepest common ancestor, the shortest path through a common ancestor, whether the two concepts share a
descendant and whether c is a parent of a child of g. A code that is not usable at T is mapped to the targets of
its active historical associations at T (all association reference sets of the release), and the pair is classified
on the mapped concepts; when a mapped concept equals the reference concept, the pair is "same concept after
mapping". When several targets exist, the closest class is kept in the order: same concept after mapping, ancestor,
descendant, sibling, lateral, other top-level hierarchy. A mention with several accepted alternatives is classified by
its closest alternative in the same order. The design-weighted share of the NARROW frame is sum_h W_h p_h with
W_h = N_h / N over benchmarks h, variance sum_h W_h^2 (1 - n_h / N_h) p_h (1 - p_h) / (n_h - 1), and a Wald 95%
interval truncated at 0. The same classification is applied to every presented candidate.

Penalty. In a sampled mention, an exact-match error of a system is credited when its top-ranked prediction equals an
accepted alternative (scenario S1); in scenario S2 every error on a mention with an accepted alternative is credited,
whatever the prediction. For a cell, the credited share r is the number of credited errors divided by the number of
sampled mentions of the benchmark, and the NARROW error count a of the cell becomes max(0, a - N_NARROW * r). Per
system, r is design-weighted over the benchmarks on which the system was evaluated. The Mantel-Haenszel risk ratio over
the 14 cells is sum(a * n_EXACT / n) / sum(b * n_NARROW / n), with b the EXACT error count. Bounds: per-cell
Clopper-Pearson 95% limits of r applied to all cells at once; and 10,000 stratified resamples of the reviewed mentions
(with replacement within each benchmark, the same draw for all systems, the panel held constant; linear congruential
generator x <- (1664525 x + 1013904223) mod 2^32 with seed 20261112, a draw being floor(n * x / 2^32)), percentile
interval at positions floor(q * (B - 1)) of the sorted values. Symmetric check: EXACT errors credited at the same
relative rate as the NARROW errors of the cell (b * (1 - credited / sampled errors)). Break-even: the common relative
reduction of the NARROW error counts of all cells at which the risk ratio equals 1 (bisection).

Standard library only. Reading the Full relationship file of the International Edition takes a few minutes and about
2 GB of memory.
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import zipfile
from collections import deque

TIME_POINTS = ['2025-10-17', '2026-11-01']
BENCHMARKS = ['DisTEMIST', 'MedProcNER', 'SympTEMIST']
SYSTEMS = ['lexical', 'sapbert', 'sapbert_ft', 'llm_rerank', 'llm_gen']
Z95 = 1.959963984540054
CLASS_ORDER = ['same_concept_after_mapping', 'ancestor', 'descendant', 'sibling', 'lateral', 'other_top_level_hierarchy', 'unresolved']
SEED = 20261112
RESAMPLES = 10000


def compact(iso):
    return iso.replace('-', '')


def member(zf, fragment):
    names = [n for n in zf.namelist() if fragment in n and n.endswith('.txt')]
    if len(names) != 1:
        raise SystemExit('expected one file matching %s in %s' % (fragment, zf.filename))
    return names[0]


def rows(zf, fragment):
    with zf.open(member(zf, fragment)) as raw:
        f = io.TextIOWrapper(raw, encoding='utf-8', newline='')
        next(f)
        for line in f:
            yield line.rstrip('\r\n').split('\t')


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def codes(s):
    out = []
    for t in s.split(';'):
        t = t.strip()
        if t and t not in out:
            out.append(t)
    return out


# ------------------------------------------------------------------ release files
def load_release(path, isa, inferred):
    cuts = [compact(t) for t in TIME_POINTS]
    zf = zipfile.ZipFile(path)
    rel = [dict() for _ in cuts]
    for r in rows(zf, 'sct2_Relationship_Full_'):
        if r[7] != isa or r[8] != inferred:
            continue
        for k, cut in enumerate(cuts):
            if r[1] <= cut:
                old = rel[k].get(r[0])
                if old is None or old[0] < r[1]:
                    rel[k][r[0]] = (r[1], r[2], r[4], r[5])
    con = [dict() for _ in cuts]
    for r in rows(zf, 'sct2_Concept_Full_'):
        for k, cut in enumerate(cuts):
            if r[1] <= cut:
                old = con[k].get(r[0])
                if old is None or old[0] < r[1]:
                    con[k][r[0]] = (r[1], r[2])
    assoc = [dict() for _ in cuts]
    for r in rows(zf, 'der2_cRefset_AssociationFull_'):
        for k, cut in enumerate(cuts):
            if r[1] <= cut:
                old = assoc[k].get(r[0])
                if old is None or old[0] < r[1]:
                    assoc[k][r[0]] = (r[1], r[2], r[5], r[6])
    graphs = []
    for k in range(len(cuts)):
        parents, edges = {}, 0
        for _, active, src, dst in rel[k].values():
            if active == '1':
                parents.setdefault(src, []).append(dst)
                edges += 1
        active = {c for c, (_, a) in con[k].items() if a == '1'}
        targets = {}
        for _, a, source, target in assoc[k].values():
            if a == '1':
                targets.setdefault(source, []).append(target)
        graphs.append(Graph(parents, active, targets, edges))
    return graphs


class Graph:
    def __init__(self, parents, active, targets, edges):
        self.P, self.active, self.targets, self.edges = parents, active, targets, edges
        self.C = {}
        for s, ps in parents.items():
            for p in ps:
                self.C.setdefault(p, []).append(s)
        roots = {p for ps in parents.values() for p in ps} - set(parents)
        if len(roots) != 1:
            raise SystemExit('the is-a hierarchy must have exactly one root, found %d' % len(roots))
        self.root = roots.pop()
        self._depth = {self.root: 0}

    def usable(self, c):
        return c in self.active and bool(self.P.get(c))

    def depth(self, x):
        if x in self._depth:
            return self._depth[x]
        stack = [x]
        while stack:
            y = stack[-1]
            pending = [p for p in self.P.get(y, []) if p not in self._depth]
            if pending:
                stack.extend(pending)
                continue
            ps = self.P.get(y, [])
            self._depth[y] = 1 + max(self._depth[p] for p in ps) if ps else float('nan')
            stack.pop()
        return self._depth[x]

    def up(self, x):
        d = {x: 0}
        frontier = [x]
        while frontier:
            nxt = []
            for y in frontier:
                for p in self.P.get(y, []):
                    if p not in d:
                        d[p] = d[y] + 1
                        nxt.append(p)
            frontier = nxt
        return d

    def down(self, x):
        seen, frontier = set(), [x]
        while frontier:
            nxt = []
            for y in frontier:
                for c in self.C.get(y, []):
                    if c not in seen:
                        seen.add(c)
                        nxt.append(c)
            frontier = nxt
        return seen

    def top(self, x):
        return [a for a in self.up(x) if self.root in self.P.get(a, [])]

    def relate(self, gold, c):
        ug, uc = self.up(gold), self.up(c)
        if c in ug:
            return {'cls': 'ancestor', 'hops': ug[c]}
        if gold in uc:
            return {'cls': 'descendant', 'hops': uc[gold]}
        lca, ld, min_hop = None, -1, float('inf')
        for a, dg in ug.items():
            if a in uc:
                d = self.depth(a)
                if d > ld:
                    ld, lca = d, a
                min_hop = min(min_hop, dg + uc[a])
        pc = set(self.P.get(c, []))
        sib = any(p in pc for p in self.P.get(gold, []))
        dg_set = self.down(gold)
        common = bool(dg_set) and not dg_set.isdisjoint(self.down(c))
        co_parent = any(c in self.P.get(x, []) for x in self.C.get(gold, []))
        cls = 'sibling' if sib else 'other_top_level_hierarchy' if lca == self.root else 'lateral'
        return {'cls': cls, 'lca_depth': ld, 'min_hops': min_hop, 'common_descendant': common, 'direct_co_parent': co_parent}

    def resolve(self, c):
        if self.usable(c):
            return [c]
        return [t for t in self.targets.get(c, []) if self.usable(t)]

    def classify(self, gold, c):
        """Closest relation of c to gold after mapping codes that are not usable to their association targets."""
        if self.usable(gold) and self.usable(c):
            r = self.relate(gold, c)
            r['route'] = 'direct'
            return r
        gs, cs = self.resolve(gold), self.resolve(c)
        if not gs or not cs:
            return {'cls': 'unresolved', 'route': 'none'}
        out = []
        for x in gs:
            for y in cs:
                out.append({'cls': 'same_concept_after_mapping'} if x == y else self.relate(x, y))
        out.sort(key=lambda r: CLASS_ORDER.index(r['cls']))
        r = out[0]
        r['route'] = 'association'
        return r


# ------------------------------------------------------------------ statistics
def weighted(frame, get):
    present = [b for b in BENCHMARKS if get(b) is not None]
    nf = sum(frame[b] for b in present)
    p = v = 0.0
    for b in present:
        k, n = get(b)
        ph, w = k / n, frame[b] / nf
        p += w * ph
        v += w * w * (1 - n / frame[b]) * ph * (1 - ph) / (n - 1)
    se = math.sqrt(v)
    return p, se, [max(0.0, p - Z95 * se), p + Z95 * se]


def lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def betacf(x, a, b):
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1 / (1 + aa * d)
        c = 1 + aa / c
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1 / (1 + aa * d)
        c = 1 + aa / c
        h *= d * c
    return h


def ibeta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta(a, b))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(x, a, b) / a
    return 1 - bt * betacf(1 - x, b, a) / b


def cp_upper(k, n):
    if k == n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):
        m = (lo + hi) / 2
        if 1 - ibeta(m, k + 1, n - k) > 0.025:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def cp_lower(k, n):
    if k == 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(100):
        m = (lo + hi) / 2
        if ibeta(m, k, n - k + 1) < 0.025:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def mh(cells, a_of, b_of=lambda c: c['b']):
    num = den = 0.0
    for c in cells:  # plain left-to-right sums, identical on every Python version
        num += a_of(c) * c['n0'] / c['N']
        den += b_of(c) * c['n1'] / c['N']
    return num / den


def tally(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--international', required=True)
    ap.add_argument('--review', required=True)
    ap.add_argument('--isa-type-id', required=True)
    ap.add_argument('--inferred-characteristic-id', required=True)
    ap.add_argument('--data', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    review = read_csv(a.review)
    for r in review:
        r['accepted'] = codes(r['accepted_codes'])
        r['candidates'] = codes(r['candidate_codes'])
        for c in r['accepted']:
            if c not in r['candidates']:
                raise SystemExit('an accepted code is not among the presented candidates')
    sample = {b: [r for r in review if r['benchmark'] == b] for b in BENCHMARKS}

    # panel counts of the 14 gold-inclusive cells
    panel = {r['item_id']: r for r in read_csv(os.path.join(a.data, 'panel_items.csv'))}
    frame = {b: sum(1 for r in panel.values() if r['benchmark'] == b and r['label'] == 'NARROW') for b in BENCHMARKS}
    acc = {}
    for s in read_csv(os.path.join(a.data, 'mention_scores.csv')):
        if s['candidate_space'] != 'oracle':
            continue
        it = panel[s['item_id']]
        c = acc.setdefault((it['benchmark'], s['system']), {'bench': it['benchmark'], 'sys': s['system'], 'n1': 0, 'a': 0, 'n0': 0, 'b': 0})
        if it['label'] == 'NARROW':
            c['n1'] += 1
            c['a'] += s['exact'] == '0'
        else:
            c['n0'] += 1
            c['b'] += s['exact'] == '0'
    cells = [acc[k] for k in sorted(acc, key=lambda k: (BENCHMARKS.index(k[0]), SYSTEMS.index(k[1])))]
    for c in cells:
        c['N'] = c['n0'] + c['n1']
    if len(cells) != 14:
        raise SystemExit('expected 14 gold-inclusive cells')

    # agreement of the review file with the judgments of the package
    jud = read_csv(os.path.join(a.data, 'alternative_code_judgments.csv'))
    cset = lambda s: frozenset(t for t in s.split(';') if t)
    final = {}
    for r in jud:
        disagree = r['rater_A_decision'] != r['rater_B_decision'] or (r['rater_A_decision'] == 'selected' and cset(r['rater_A_candidates']) != cset(r['rater_B_candidates']))
        decision = r['consensus_decision'] if disagree else r['rater_A_decision']
        final[r['benchmark']] = final.get(r['benchmark'], 0) + (decision == 'selected')
    for b in BENCHMARKS:
        if final.get(b, 0) != sum(1 for r in sample[b] if r['accepted']) or len(sample[b]) != sum(1 for r in jud if r['benchmark'] == b):
            raise SystemExit('the review file does not match data/alternative_code_judgments.csv for ' + b)

    out = {'reviewed_mentions': len(review), 'frame_NARROW': sum(frame.values())}
    for b in BENCHMARKS:
        out['reviewed_mentions_' + b] = len(sample[b])
        out['mentions_with_accepted_alternative_' + b] = sum(1 for r in sample[b] if r['accepted'])
        out['frame_NARROW_' + b] = frame[b]
    out['mentions_with_accepted_alternative'] = sum(1 for r in review if r['accepted'])
    out['accepted_alternative_pairs'] = sum(len(r['accepted']) for r in review)
    out['risk_ratio_unadjusted'] = mh(cells, lambda c: c['a'])

    # (A) hierarchy
    graphs = load_release(a.international, a.isa_type_id, a.inferred_characteristic_id)
    for t, g in zip(TIME_POINTS, graphs):
        pairs, cand, closest = [], [], []
        for r in review:
            mine = []
            for c in r['accepted']:
                x = g.classify(r['gold_code'], c)
                pairs.append(x)
                mine.append(x['cls'])
            if mine:
                closest.append(min(mine, key=CLASS_ORDER.index))
            for c in r['candidates']:
                cand.append((g.classify(r['gold_code'], c), c in r['accepted'], c == r['gold_code']))
        lat = [p for p in pairs if p['cls'] in ('sibling', 'lateral')]
        res = {
            'active_isa_relationships': g.edges,
            'pairs_by_class': tally(p['cls'] for p in pairs),
            'pairs_by_route': tally(p['route'] for p in pairs),
            'ancestor_pairs_by_hops': tally(p['hops'] for p in pairs if p['cls'] == 'ancestor'),
            'descendant_pairs_by_hops': tally(p['hops'] for p in pairs if p['cls'] == 'descendant'),
            'mentions_by_closest_class': tally(closest),
            'sibling_or_lateral_pairs': len(lat),
            'sibling_or_lateral_pairs_by_common_ancestor_depth': tally(p['lca_depth'] for p in lat),
            'sibling_or_lateral_pairs_by_shortest_path': tally(p['min_hops'] for p in lat),
            'sibling_or_lateral_pairs_sharing_a_descendant': sum(p['common_descendant'] for p in lat),
            'sibling_or_lateral_pairs_alternative_is_parent_of_a_child': sum(p['direct_co_parent'] for p in lat),
            'presented_candidates': len(cand),
            'presented_candidates_equal_to_gold': sum(1 for x in cand if x[2]),
            'presented_and_accepted_by_class': {k: {'presented': sum(1 for x in cand if x[0]['cls'] == k), 'accepted': sum(1 for x in cand if x[0]['cls'] == k and x[1])}
                                                for k in CLASS_ORDER if any(x[0]['cls'] == k for x in cand)},
        }
        anc = [r for r in review if any(g.classify(r['gold_code'], c)['cls'] == 'ancestor' for c in r['candidates'])]
        res['mentions_with_an_ancestor_among_candidates'] = len(anc)
        res['mentions_with_an_accepted_ancestor_among_them'] = sum(1 for r in anc if any(g.classify(r['gold_code'], c)['cls'] == 'ancestor' for c in r['accepted']))
        res['mentions_with_a_direct_parent_among_candidates'] = sum(1 for r in review if any(
            (lambda x: x['cls'] == 'ancestor' and x['hops'] == 1)(g.classify(r['gold_code'], c)) for c in r['candidates']))
        if t == TIME_POINTS[0]:
            def share(pred):
                cls = {}
                for r in review:
                    if r['accepted']:
                        cls[id(r)] = min((g.classify(r['gold_code'], c)['cls'] for c in r['accepted']), key=CLASS_ORDER.index)
                p, se, ci = weighted(frame, lambda b: (sum(1 for r in sample[b] if id(r) in cls and pred(cls[id(r)])), len(sample[b])))
                return {'mentions': sum(1 for v in cls.values() if pred(v)), 'weighted_share': p, 'ci95_low': ci[0], 'ci95_high': ci[1]}
            res['weighted_by_closest_class'] = {k: share(lambda v, k=k: v == k) for k in CLASS_ORDER[:-1]}
            res['weighted_by_closest_class']['sibling_or_lateral'] = share(lambda v: v in ('sibling', 'lateral'))
            res['weighted_by_closest_class']['any_accepted_alternative'] = share(lambda v: True)
        out['hierarchy_' + t] = res
    out['hierarchy_classes_identical_at_both_time_points'] = all(
        out['hierarchy_' + TIME_POINTS[0]][k] == out['hierarchy_' + TIME_POINTS[1]][k]
        for k in ('pairs_by_class', 'pairs_by_route', 'mentions_by_closest_class', 'sibling_or_lateral_pairs_sharing_a_descendant'))

    # (B) penalty
    for c in cells:
        s = sample[c['bench']]
        err = credited = on_alt = in_cand = 0
        for r in s:
            p = r['predicted_' + c['sys']]
            if not p:
                raise SystemExit('missing prediction for an evaluated cell')
            if p == r['gold_code']:
                continue
            err += 1
            credited += p in r['accepted']
            on_alt += bool(r['accepted'])
            in_cand += p in r['candidates']
        c.update(sN=len(s), sErr=err, sResc=credited, sAlt=on_alt, sCand=in_cand)
    pen = {'sampled_errors': sum(c['sErr'] for c in cells), 'sampled_errors_with_prediction_among_candidates': sum(c['sCand'] for c in cells),
           'sampled_errors_credited_S1': sum(c['sResc'] for c in cells), 'sampled_errors_on_mentions_with_accepted_alternative': sum(c['sAlt'] for c in cells),
           'per_system': {}}
    for sy in SYSTEMS:
        cs = {c['bench']: c for c in cells if c['sys'] == sy}
        get = lambda key: (lambda b: (cs[b][key], cs[b]['sN']) if b in cs else None)
        fr = sum(c['n1'] for c in cs.values())
        fe = sum(c['a'] for c in cs.values())
        p1, se1, ci1 = weighted(frame, get('sResc'))
        p2, _, _ = weighted(frame, get('sAlt'))
        pen['per_system'][sy] = {
            'benchmarks': [b for b in BENCHMARKS if b in cs], 'frame_NARROW': fr, 'error_rate': fe / fr,
            'sampled_errors': sum(c['sErr'] for c in cs.values()), 'credited_S1': sum(c['sResc'] for c in cs.values()),
            'credited_share_S1': p1, 'credited_share_S1_ci95_low': ci1[0], 'credited_share_S1_ci95_high': ci1[1],
            'error_rate_S1': (fe - fr * p1) / fr, 'reduction_pp_S1': 100 * p1, 'relative_reduction_S1': p1 / (fe / fr),
            'credited_share_S2': p2, 'error_rate_S2': (fe - fr * p2) / fr, 'relative_reduction_S2': p2 / (fe / fr),
        }
    clamp = lambda c, r: max(0.0, c['a'] - c['n1'] * r)
    rr = {'unadjusted': out['risk_ratio_unadjusted']}
    rr['S1'] = mh(cells, lambda c: clamp(c, c['sResc'] / c['sN']))
    rr['S1_cp_lower_share'] = mh(cells, lambda c: clamp(c, cp_lower(c['sResc'], c['sN'])))
    rr['S1_cp_upper_share'] = mh(cells, lambda c: clamp(c, cp_upper(c['sResc'], c['sN'])))
    rr['S2'] = mh(cells, lambda c: clamp(c, c['sAlt'] / c['sN']))
    rr['S2_cp_upper_share'] = mh(cells, lambda c: clamp(c, cp_upper(c['sAlt'], c['sN'])))
    rel1 = lambda c: c['sResc'] / c['sErr'] if c['sErr'] else 0.0
    rel2 = lambda c: c['sAlt'] / c['sErr'] if c['sErr'] else 0.0
    rr['S1_symmetric_EXACT'] = mh(cells, lambda c: clamp(c, c['sResc'] / c['sN']), lambda c: c['b'] * (1 - rel1(c)))
    rr['S2_symmetric_EXACT'] = mh(cells, lambda c: clamp(c, c['sAlt'] / c['sN']), lambda c: c['b'] * (1 - rel2(c)))
    lo, hi = 0.0, 1.0
    for _ in range(100):
        m = (lo + hi) / 2
        if mh(cells, lambda c: c['a'] * (1 - m)) > 1:
            lo = m
        else:
            hi = m
    rr['relative_NARROW_error_reduction_for_ratio_1'] = (lo + hi) / 2
    state = [SEED]

    def rnd():
        state[0] = (state[0] * 1664525 + 1013904223) % 2 ** 32
        return state[0] / 2 ** 32
    b1, b2 = [], []
    for _ in range(RESAMPLES):
        pick = {}
        for b in BENCHMARKS:
            s = sample[b]
            pick[b] = [s[int(math.floor(rnd() * len(s)))] for _ in s]

        def rate(c, mode):
            k = 0
            for r in pick[c['bench']]:
                p = r['predicted_' + c['sys']]
                if p == r['gold_code']:
                    continue
                if (p in r['accepted']) if mode == 1 else bool(r['accepted']):
                    k += 1
            return k / len(pick[c['bench']])
        b1.append(mh(cells, lambda c: clamp(c, rate(c, 1))))
        b2.append(mh(cells, lambda c: clamp(c, rate(c, 2))))
    pct = lambda v, q: sorted(v)[int(math.floor(q * (len(v) - 1)))]
    rr['S1_resample95'] = [pct(b1, 0.025), pct(b1, 0.975)]
    rr['S2_resample95'] = [pct(b2, 0.025), pct(b2, 0.975)]
    pen['risk_ratio'] = rr
    out['penalty'] = pen
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write('\n')


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main()
