#!/usr/bin/env python3
"""Aggregate tables of the terminology-release analysis, computed from licensed SNOMED CT release files.

Usage:
  python3 terminology_release_tables.py --international INTERNATIONAL_FULL_RELEASE.zip \
      --spanish SPANISH_EDITION_RELEASE.zip --predictions predictions.csv \
      --isa-type-id ID --inferred-characteristic-id ID --core-module-id ID --out DIR

Inputs (none is distributed with the package):
  --international  SNOMED CT International Edition release archive (RF2) of 1 November 2026 or later, whose Full
                   files cover every time point up to 1 November 2026
  --spanish        SNOMED CT Spanish Edition release archive of 31 October 2022 (extension files only)
  --predictions    CSV, one row per scored mention of the 14 gold-inclusive cells, with columns benchmark, system,
                   label (EXACT or NARROW), gold_code and predicted_code (top-ranked prediction)
  --isa-type-id, --inferred-characteristic-id, --core-module-id
                   identifiers of three SNOMED CT metadata concepts: the is-a relationship type, the inferred
                   characteristic type and the International core module. They are not written in this file; take
                   them from the documentation of the SNOMED CT release you use.
Outputs (in --out): time_points.csv, cells.csv, gold_states_by_benchmark.csv, compatible_releases.csv and
summary.csv, with the columns of the files in data/terminology_release. Release dates are written as YYYY-MM-DD.

Method. For an International time point T, every component takes its latest version with effectiveTime <= T. The
inferred hierarchy consists of the active inferred is-a relationships; the stated hierarchy of the active OWL axioms
of the form SubClassOf(:child :parent) (each axiom identified by its reference-set member identifier, so a later
version that is no longer of that form removes the edge). The evaluated concepts are all gold and predicted codes.
Closure edges are the is-a edges between concepts of the ancestor closure of the evaluated concepts. A gold concept
is absent at T if it has no version up to T, otherwise active or inactive. Depth is the longest is-a path to the root.
The Wu-Palmer score of a prediction p for gold g is 1 if p = g, 0 if p or g has no active parent, and otherwise
2 * depth(deepest common ancestor) / (depth(p) + depth(g)). The partial-credit gap of a row is its score minus its
exact-match indicator; the credit-gap ratio of a cell is the mean gap of its NARROW rows divided by that of its EXACT
rows, and the time-point value is the geometric mean over the cells. A row has changed when its score, written with
four decimals, differs from that at the reference time point. The Spanish-edition time point is the International
release named in the module dependency reference set of the Spanish release, with the extension concepts,
relationships and axioms (latest version per identifier up to the Spanish release date) added; gold states are those
of that International release. A release is compatible with a benchmark when all its gold concepts exist in it, and
fully compatible when all are also active; releases are the distinct effective times of the International concept file.

Standard library only. Reading the Full relationship and OWL files of the International Edition needs about 2.5 GB
of memory and well under a minute.
"""
import argparse
import collections
import csv
import functools
import io
import math
import os
import re
import sys
import zipfile



def compact(iso):
    """YYYY-MM-DD -> YYYYMMDD, the effectiveTime format of the release files."""
    return iso.replace('-', '')


def iso(date):
    """YYYYMMDD -> YYYY-MM-DD, the date format of the output tables."""
    return '%s-%s-%s' % (date[:4], date[4:6], date[6:]) if date else ''


TIME_POINTS = [compact(t) for t in ('2021-07-31', '2021-10-21', '2022-10-31', '2022-12-31', '2023-12-31', '2025-10-17', '2026-11-01')]
REFERENCE = compact('2025-10-17')
SPANISH_RELEASE = compact('2022-10-31')
SUBCLASS = re.compile(r'^SubClassOf\(:(\d+)\s+:(\d+)\)\s*$')
BENCHMARKS = ['DisTEMIST', 'MedProcNER', 'SympTEMIST']
sys.setrecursionlimit(100000)


def member(zf, fragment):
    names = [n for n in zf.namelist() if fragment in n]
    if len(names) != 1:
        raise SystemExit('expected one file matching %s in %s' % (fragment, zf.filename))
    return names[0]


def rows(zf, fragment):
    with zf.open(member(zf, fragment)) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig'), delimiter='\t', quoting=csv.QUOTE_NONE)


def latest(versions, date):
    """versions: list of (effective_time, ...) sorted by effective time; the last one not after date, or None."""
    best = None
    for v in versions:
        if v[0] <= date:
            best = v
        else:
            break
    return best


def parents_at(rel_versions, date):
    parents = collections.defaultdict(set)
    for versions in rel_versions.values():
        v = latest(versions, date)
        if v is not None and v[1] == '1' and v[2] is not None:
            parents[v[2]].add(v[3])
    return parents


def closure_edges(parents, concepts):
    seen, stack = set(), list(concepts)
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        stack.extend(parents.get(c, ()))
    return {(s, t) for s in seen for t in parents.get(s, ())}


def scorer(parents):
    @functools.lru_cache(maxsize=None)
    def ancestry(c):
        anc, depth = {c}, 0
        for p in parents.get(c, ()):
            a, d = ancestry(p)
            anc |= a
            depth = max(depth, d + 1)
        return frozenset(anc), depth

    def wup(p, g):
        if p == g:
            return 1.0
        if p not in parents or g not in parents:
            return 0.0
        ap, dp = ancestry(p)
        ag, dg = ancestry(g)
        common = ap & ag
        if not common or dp + dg == 0:
            return 0.0
        return 2 * max(ancestry(c)[1] for c in common) / (dp + dg)
    return wup


def score_cells(cells, parents):
    wup = scorer(parents)
    return {k: [wup(r['predicted_code'], r['gold_code']) for r in rs] for k, rs in cells.items()}


def cell_ratios(cells, scores):
    out = {}
    for k, rs in cells.items():
        gap = {'EXACT': [], 'NARROW': []}
        for r, s in zip(rs, scores[k]):
            gap[r['label']].append(s - (r['predicted_code'] == r['gold_code']))
        out[k] = (len(gap['EXACT']), len(gap['NARROW']), math.fsum(gap['EXACT']) / len(gap['EXACT']), math.fsum(gap['NARROW']) / len(gap['NARROW']))
    return out


def compare(cells, ref, other):
    per_cell, up, down = {}, 0, 0
    for k in cells:
        n = 0
        for a, b in zip(ref[k], other[k]):
            if '%.4f' % a != '%.4f' % b:
                n += 1
                up += b > a
                down += b < a
        per_cell[k] = n
    return per_cell, up, down


def states(history, concepts, date):
    c = collections.Counter()
    for g in concepts:
        v = latest(history.get(g, []), date)
        c['absent' if v is None else ('active' if v[1] == '1' else 'inactive')] += 1
    return c['active'], c['inactive'], c['absent']


def write(path, header, table):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--international', required=True)
    ap.add_argument('--spanish', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--isa-type-id', required=True, help='identifier of the is-a relationship type')
    ap.add_argument('--inferred-characteristic-id', required=True, help='identifier of the inferred characteristic type')
    ap.add_argument('--core-module-id', required=True, help='identifier of the International core module')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    IS_A, INFERRED, CORE_MODULE = a.isa_type_id, a.inferred_characteristic_id, a.core_module_id
    os.makedirs(a.out, exist_ok=True)

    with open(a.predictions, encoding='utf-8', newline='') as f:
        pred = list(csv.DictReader(f))
    cells = collections.OrderedDict()
    for r in sorted(pred, key=lambda r: (BENCHMARKS.index(r['benchmark']), r['system'])):
        cells.setdefault((r['benchmark'], r['system']), []).append(r)
    gold = {r['gold_code'] for r in pred}
    predicted = {r['predicted_code'] for r in pred}
    evaluated = gold | predicted
    gold_by = {b: {r['gold_code'] for r in pred if r['benchmark'] == b} for b in BENCHMARKS}

    zi = zipfile.ZipFile(a.international)
    history, releases = collections.defaultdict(list), set()
    for r in rows(zi, '/Full/Terminology/sct2_Concept_Full_'):
        releases.add(r['effectiveTime'])
        if r['id'] in gold:
            history[r['id']].append((r['effectiveTime'], r['active']))
    if max(releases) < max(TIME_POINTS):
        raise SystemExit('the International release must cover %s; latest release in the archive is %s' % (iso(max(TIME_POINTS)), iso(max(releases))))
    rel = collections.defaultdict(list)
    for r in rows(zi, '/Full/Terminology/sct2_Relationship_Full_'):
        if r['typeId'] == IS_A and r['characteristicTypeId'] == INFERRED:
            rel[r['id']].append((r['effectiveTime'], r['active'], r['sourceId'], r['destinationId']))
    owl = collections.defaultdict(list)
    for r in rows(zi, '/Full/Terminology/sct2_sRefset_OWLExpressionFull_'):
        m = SUBCLASS.match(r['owlExpression'] or '')
        owl[r['id']].append((r['effectiveTime'], r['active'], m.group(1) if m else None, m.group(2) if m else None))
    for d in (history, rel, owl):
        for v in d.values():
            v.sort()

    # Spanish extension: latest version per identifier up to the Spanish release date
    zs = zipfile.ZipFile(a.spanish)
    dep = [r for r in rows(zs, '/Snapshot/Refset/Metadata/der2_ssRefset_ModuleDependency')
           if r['referencedComponentId'] == CORE_MODULE and r['active'] == '1' and r['sourceEffectiveTime'] == SPANISH_RELEASE]
    if len(dep) != 1:
        raise SystemExit('the International release declared by the Spanish release was not found')
    base = dep[0]['targetEffectiveTime']
    ext_rel, ext_owl = {}, {}
    for r in rows(zs, '/Full/Terminology/sct2_Relationship_'):
        if r['typeId'] == IS_A and r['characteristicTypeId'] == INFERRED and r['effectiveTime'] <= SPANISH_RELEASE:
            v = (r['effectiveTime'], r['active'], r['sourceId'], r['destinationId'])
            if r['id'] not in ext_rel or v[0] > ext_rel[r['id']][0]:
                ext_rel[r['id']] = v
    for r in rows(zs, '/Full/Terminology/sct2_sRefset_OWLExpression'):
        if r['effectiveTime'] <= SPANISH_RELEASE:
            m = SUBCLASS.match(r['owlExpression'] or '')
            v = (r['effectiveTime'], r['active'], m.group(1) if m else None, m.group(2) if m else None)
            if r['id'] not in ext_owl or v[0] > ext_owl[r['id']][0]:
                ext_owl[r['id']] = v
    ext_concepts = sorted({r['id'] for r in rows(zs, '/Snapshot/Terminology/sct2_Concept_')})
    ext_isa = [r for r in rows(zs, '/Snapshot/Terminology/sct2_Relationship_') if r['typeId'] == IS_A]
    ext_endpoints = {r['sourceId'] for r in ext_isa} | {r['destinationId'] for r in ext_isa} | set(ext_concepts)

    hierarchies, stated = {}, {}
    for t in TIME_POINTS + [base]:
        hierarchies[t] = parents_at(rel, t)
        stated[t] = parents_at(owl, t)
    spanish_rel = {k: [latest(v, base)] for k, v in rel.items() if latest(v, base) is not None}
    spanish_rel.update({k: [v] for k, v in ext_rel.items()})
    spanish_owl = {k: [latest(v, base)] for k, v in owl.items() if latest(v, base) is not None}
    spanish_owl.update({k: [v] for k, v in ext_owl.items()})
    hierarchies['spanish'] = parents_at(spanish_rel, '99999999')
    stated['spanish'] = parents_at(spanish_owl, '99999999')

    scores = {t: score_cells(cells, h) for t, h in hierarchies.items()}
    inferred_edges = {t: closure_edges(h, evaluated) for t, h in hierarchies.items()}
    stated_edges = {t: closure_edges(h, evaluated) for t, h in stated.items()}

    points, cell_rows = [], []
    for t in TIME_POINTS + ['spanish']:
        ratios = cell_ratios(cells, scores[t])
        per_cell, up, down = compare(cells, scores[REFERENCE], scores[t])
        gm = math.exp(math.fsum(math.log(x[3] / x[2]) for x in ratios.values()) / len(ratios))
        g = states(history, gold, base if t == 'spanish' else t)
        name, release = ('spanish_edition', iso(base)) if t == 'spanish' else ('international_' + iso(t), iso(t))
        ie, se = inferred_edges[t], stated_edges[t]
        points.append([name, release, len(ie), len(inferred_edges[REFERENCE] - ie), len(ie - inferred_edges[REFERENCE]), len(se),
                       len(stated_edges[REFERENCE] - se), len(se - stated_edges[REFERENCE]), g[0], g[1], g[2], sum(per_cell.values()),
                       up, down, sum(v > 0 for v in per_cell.values()), repr(gm)])
        if t != 'spanish':
            for (b, s), (ne, nn, ge, gn) in ratios.items():
                cell_rows.append([name, b, s, ne, nn, repr(ge), repr(gn), per_cell[(b, s)]])
    write(os.path.join(a.out, 'time_points.csv'),
          ['time_point', 'international_release', 'inferred_edges', 'inferred_edges_added_vs_reference', 'inferred_edges_deleted_vs_reference',
           'stated_edges', 'stated_edges_added_vs_reference', 'stated_edges_deleted_vs_reference', 'gold_active', 'gold_inactive', 'gold_absent',
           'rows_changed_vs_reference', 'rows_up', 'rows_down', 'cells_with_changed_rows', 'credit_gap_ratio_geometric_mean'], points)
    write(os.path.join(a.out, 'cells.csv'), ['time_point', 'benchmark', 'system', 'exact_rows', 'narrow_rows', 'credit_gap_exact',
                                             'credit_gap_narrow', 'rows_changed_vs_reference'], cell_rows)
    write(os.path.join(a.out, 'gold_states_by_benchmark.csv'), ['time_point', 'benchmark', 'gold_concepts', 'active', 'inactive', 'absent'],
          [['international_' + iso(t), b, len(gold_by[b])] + list(states(history, gold_by[b], t)) for t in TIME_POINTS for b in BENCHMARKS])

    window = []
    for b in BENCHMARKS:
        series = [(r, states(history, gold_by[b], r)) for r in sorted(releases)]
        exist = [r for r, s in series if s[2] == 0]
        active = [r for r, s in series if s[2] == 0 and s[1] == 0]
        window.append([b, len(gold_by[b]), iso(exist[0]) if exist else '', iso(active[0]) if active else '', iso(active[-1]) if active else '', len(active),
                       states(history, gold_by[b], SPANISH_RELEASE)[1]])
    write(os.path.join(a.out, 'compatible_releases.csv'), ['benchmark', 'gold_concepts', 'first_release_all_exist', 'first_release_all_active',
                                                           'last_release_all_active', 'releases_all_exist_and_active', 'gold_inactive_at_20221031'], window)

    per_cell, _, _ = compare(cells, scores[base], scores['spanish'])
    write(os.path.join(a.out, 'summary.csv'), ['quantity', 'value'], [
        ['reference_release', iso(REFERENCE)], ['prediction_rows', len(pred)], ['evaluated_concepts', len(evaluated)], ['gold_concepts', len(gold)],
        ['predicted_concepts', len(predicted)], ['international_releases_in_concept_history', len(releases)], ['spanish_release', iso(SPANISH_RELEASE)],
        ['spanish_declared_international_base', iso(base)], ['spanish_extension_concepts', len(ext_concepts)], ['spanish_extension_is_a_rows', len(ext_isa)],
        ['spanish_extension_active_is_a_rows', sum(r['active'] == '1' for r in ext_isa)],
        ['spanish_extension_inactive_is_a_rows', sum(r['active'] == '0' for r in ext_isa)],
        ['spanish_extension_concepts_among_evaluated', len(ext_endpoints & evaluated)],
        ['spanish_extension_rows_changed_when_added_to_base', sum(per_cell.values())]])


if __name__ == '__main__':
    main()
