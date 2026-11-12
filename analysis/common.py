"""Shared helpers: CSV input, the Mantel-Haenszel risk ratio, Wilson intervals, kappa and percentile intervals."""
import csv
import math
import os

BENCHMARKS = ['DisTEMIST', 'MedProcNER', 'SympTEMIST']
Z95 = 1.959963984540054


def plain_sum(values):
    """Left-to-right floating-point sum (the built-in sum compensates rounding from Python 3.12 on)."""
    total = 0.0
    for v in values:
        total += v
    return total


def read_csv(data_dir, name):
    with open(os.path.join(data_dir, name), encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def load_panel(data_dir):
    """Fixed panel: items in file order (the order decides ties), and the 0/1 error of each gold-inclusive cell.

    The errors are read from the rows of mention_scores.csv whose candidate_space is oracle (gold-inclusive).

    Returns (items, cells). items: list of dicts with benchmark, label, document. cells: list of dicts with
    benchmark, system, idx (item indices) and err (0/1 errors), ordered by benchmark and system.
    """
    items = read_csv(data_dir, 'panel_items.csv')
    index = {r['item_id']: i for i, r in enumerate(items)}
    by_cell = {}
    for r in read_csv(data_dir, 'mention_scores.csv'):
        if r['candidate_space'] != 'oracle':
            continue
        i = index[r['item_id']]
        key = (items[i]['benchmark'], r['system'])
        c = by_cell.setdefault(key, {'benchmark': key[0], 'system': key[1], 'idx': [], 'err': []})
        c['idx'].append(i)
        c['err'].append(1 - int(r['exact']))
    cells = [by_cell[k] for k in sorted(by_cell, key=lambda k: (BENCHMARKS.index(k[0]), k[1]))]
    return items, index, cells


def cell_table(cell, narrow):
    """Counts (nE, eE, nN, eN) of one cell for a 0/1 NARROW indicator per item."""
    nE = eE = nN = eN = 0
    for i, e in zip(cell['idx'], cell['err']):
        if narrow[i]:
            nN += 1
            eN += e
        else:
            nE += 1
            eE += e
    return nE, eE, nN, eN


def mh_ratio_from_tables(tables):
    """Mantel-Haenszel error risk ratio, NARROW over EXACT; tables are (nE, eE, nN, eN); empty strata skipped."""
    num = den = 0.0
    for nE, eE, nN, eN in tables:
        if nE and nN:
            n = nE + nN
            num += eN * nE / n
            den += eE * nN / n
    return (num / den if den else float('nan')), num, den


def mh_ratio(cells, narrow):
    return mh_ratio_from_tables([cell_table(c, narrow) for c in cells])[0]


def wilson_n(p, n):
    """Wilson score interval for a proportion p with (possibly effective) sample size n: [p, low, high]."""
    if n <= 0:
        return [p, None, None]
    d = 1 + Z95 * Z95 / n
    c = (p + Z95 * Z95 / (2 * n)) / d
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]


def wilson(k, n):
    return wilson_n(k / n, n) if n > 0 else [None, None, None]


def kappa(pairs, categories):
    """Cohen's kappa for a list of (rating 1, rating 2): returns n, observed agreement and kappa."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = plain_sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in categories)
    return {'n': n, 'agreement': po, 'kappa': (po - pe) / (1 - pe) if pe < 1 else None}


def percentile_interval(values):
    """Elements at positions floor(0.025 m) and floor(0.975 m) of the sorted finite values (m of them)."""
    s = sorted(v for v in values if v is not None and math.isfinite(v))
    if not s:
        return [None, None]
    pick = lambda q: s[min(len(s) - 1, max(0, int(math.floor(q * len(s)))))]
    return [pick(0.025), pick(0.975)]
