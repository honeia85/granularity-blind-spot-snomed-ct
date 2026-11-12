"""Sensitivity of the NARROW/EXACT error risk ratio of the fixed panel to the granularity labels.

Risk ratio: Mantel-Haenszel error risk ratio, NARROW over EXACT, over the 14 gold-inclusive cells (benchmark x system).

Exact label budgets. With L the least common multiple of the three benchmark sizes, item i receives the integer
weight h_i = sum over the cells k of its benchmark of (err_ik * n_k - t_k) * L / n_k (n_k items and t_k errors in
cell k). For a label vector, the sum of h_i over NARROW items equals L times (numerator - denominator) of the
Mantel-Haenszel ratio, so the ratio is at most 1 exactly when that sum is at most 0. Within a benchmark, relabeling
r NARROW items as EXACT and s EXACT items as NARROW reduces the sum most when the r NARROW items with the largest
weights and the s EXACT items with the smallest weights are chosen (ties broken by item order). The best reduction
for each cost r + s is combined across benchmarks by max-plus convolution, which gives the smallest number of label
changes that brings the ratio to 1 or below. Modes: any changes, NARROW to EXACT only, EXACT to NARROW only, pairs
(r = s within a benchmark, cost counted in pairs), and changes restricted to one benchmark. Every benchmark keeps at
least one item in each label group.

Non-differential relabeling: with probabilities p (EXACT to NARROW) and q (NARROW to EXACT) the expected cell
tables give an expected-cell ratio whose numerator minus denominator equals (1 - p - q) times its baseline value;
the grid p, q in {0, 0.01, ..., 1} is evaluated (the two points at which one label group is empty are skipped).

Linked items: panel items with an exact link to mentions that carry three independent labels. The ratio is
computed within the linked MedProcNER items whose three labels agree, with the panel labels and with the agreed
labels, and for the full panel with the agreed labels substituted.
"""
import math

from common import BENCHMARKS, cell_table, load_panel, mh_ratio, mh_ratio_from_tables, read_csv

NEG = None  # infeasible cost


def integer_weights(items, cells):
    sizes = [sum(1 for it in items if it['benchmark'] == b) for b in BENCHMARKS]
    L = 1
    for s in sizes:
        L = L * s // math.gcd(L, s)
    h = [0] * len(items)
    for c in cells:
        n, t = len(c['idx']), sum(c['err'])
        if n != sizes[BENCHMARKS.index(c['benchmark'])]:
            raise ValueError('a cell does not cover all items of its benchmark')
        w = L // n
        for i, e in zip(c['idx'], c['err']):
            h[i] += (e * n - t) * w
    return h, L


def benchmark_lists(h, narrow, bench_of):
    lists = []
    for b in range(len(BENCHMARKS)):
        N = sorted((i for i in range(len(h)) if bench_of[i] == b and narrow[i]), key=lambda i: (-h[i], i))
        E = sorted((i for i in range(len(h)) if bench_of[i] == b and not narrow[i]), key=lambda i: (h[i], i))
        pN, pE = [0], [0]
        for i in N:
            pN.append(pN[-1] + h[i])
        for i in E:
            pE.append(pE[-1] + h[i])
        lists.append({'N': N, 'E': E, 'pN': pN, 'pE': pE, 'nN': len(N), 'n': len(N) + len(E)})
    return lists


def benchmark_curve(bl, mode):
    nN, n, pN, pE = bl['nN'], bl['n'], bl['pN'], bl['pE']
    top = {'pairs': min(nN, n - nN), 'NtoE': nN, 'EtoN': n - nN}.get(mode, n)
    best, arg = [NEG] * (top + 1), [None] * (top + 1)
    ok = lambda r, s: 1 <= nN - r + s <= n - 1
    if mode == 'pairs':
        for r in range(top + 1):
            if ok(r, r):
                best[r], arg[r] = pN[r] - pE[r], (r, r)
        return best, arg
    r_max = 0 if mode == 'EtoN' else nN
    s_max = 0 if mode == 'NtoE' else n - nN
    for r in range(r_max + 1):
        for s in range(s_max + 1):
            if not ok(r, s):
                continue
            v, m = pN[r] - pE[s], r + s
            if best[m] is NEG or v > best[m] or (v == best[m] and r < arg[m][0]):
                best[m], arg[m] = v, (r, s)
    return best, arg


def budget_curve(h, narrow, bench_of, mode, allowed=(0, 1, 2)):
    """Best reduction F[m] of the weighted sum for exactly m units of cost, with back-pointers to the choices."""
    lists = benchmark_lists(h, narrow, bench_of)
    acc, back = [0], []
    for b, bl in enumerate(lists):
        best, arg = benchmark_curve(bl, mode)
        if b not in allowed:
            best, arg = best[:1], arg[:1]
        nxt, ptr = [NEG] * (len(acc) + len(best) - 1), [None] * (len(acc) + len(best) - 1)
        for x, ax in enumerate(acc):
            if ax is NEG:
                continue
            for y, by in enumerate(best):
                if by is NEG:
                    continue
                if nxt[x + y] is NEG or ax + by > nxt[x + y]:
                    nxt[x + y], ptr[x + y] = ax + by, (x, arg[y])
        acc = nxt
        back.append(ptr)
    D0 = sum(h[i] for i in range(len(h)) if narrow[i])

    def assignment(m):
        v = list(narrow)
        choices = []
        for b in range(len(lists) - 1, -1, -1):
            x, rs = back[b][m]
            choices.append(rs)
            m = x
        changes = [0, 0]
        for b, (r, s) in zip(range(len(lists)), reversed(choices)):
            for i in lists[b]['N'][:r]:
                v[i] = 0
            for i in lists[b]['E'][:s]:
                v[i] = 1
            changes[0] += r
            changes[1] += s
        return v, changes

    threshold = next((m for m, f in enumerate(acc) if f is not NEG and f >= D0), None)
    feasible = max(m for m, f in enumerate(acc) if f is not NEG)
    return {'F': acc, 'threshold': threshold, 'max_feasible': feasible, 'assignment': assignment}


def run(data_dir):
    items, index, cells = load_panel(data_dir)
    bench_of = [BENCHMARKS.index(it['benchmark']) for it in items]
    narrow0 = [1 if it['label'] == 'NARROW' else 0 for it in items]
    N = len(items)
    out = {}

    # panel description
    out['items'] = N
    out['score_rows'] = sum(len(c['idx']) for c in cells)
    out['cells'] = len(cells)
    out['documents'] = len({it['document_id'] for it in items})
    out['exact_items'] = N - sum(narrow0)
    out['narrow_items'] = sum(narrow0)
    out['narrow_share'] = sum(narrow0) / N
    for b in BENCHMARKS:
        sub = [it for it in items if it['benchmark'] == b]
        out['items_' + b] = len(sub)
        out['narrow_items_' + b] = sum(1 for it in sub if it['label'] == 'NARROW')
        out['exact_items_' + b] = len(sub) - out['narrow_items_' + b]
        out['narrow_share_' + b] = out['narrow_items_' + b] / len(sub)
        out['documents_' + b] = len({it['document_id'] for it in sub})
    out['risk_ratio'] = mh_ratio(cells, narrow0)
    systems = sorted({c['system'] for c in cells})
    per_system = [mh_ratio([c for c in cells if c['system'] == s], narrow0) for s in systems]
    for s, v in zip(systems, per_system):
        out['risk_ratio_system_' + s] = v
    out['risk_ratio_system_min'] = min(per_system)
    out['risk_ratio_system_max'] = max(per_system)

    # exact label budgets
    h, L = integer_weights(items, cells)
    D0 = sum(h[i] for i in range(N) if narrow0[i])
    tables0 = [cell_table(c, narrow0) for c in cells]
    ratio, num, den = mh_ratio_from_tables(tables0)
    if abs((num - den) - D0 / L) > 1e-8:
        raise ValueError('integer weights do not reproduce the Mantel-Haenszel difference')
    modes = [('any', 'free', (0, 1, 2)), ('narrow_to_exact', 'NtoE', (0, 1, 2)), ('exact_to_narrow', 'EtoN', (0, 1, 2)),
             ('pairs', 'pairs', (0, 1, 2))] + [('only_' + b, 'free', (k,)) for k, b in enumerate(BENCHMARKS)]
    for name, mode, allowed in modes:
        cv = budget_curve(h, narrow0, bench_of, mode, allowed)
        key = 'budget_' + name
        out[key + '_max_feasible_cost'] = cv['max_feasible']
        m = cv['threshold']
        if m is None:
            out[key + '_threshold'] = None
            continue
        out[key + '_threshold'] = m
        changes = 2 * m if mode == 'pairs' else m
        out[key + '_label_changes'] = changes
        out[key + '_fraction_of_items'] = changes / N
        v, (n_to_e, e_to_n) = cv['assignment'](m)
        out[key + '_ratio_at_threshold'] = mh_ratio(cells, v)
        out[key + '_narrow_to_exact_at_threshold'] = n_to_e
        out[key + '_exact_to_narrow_at_threshold'] = e_to_n
        if m > 0:
            out[key + '_ratio_before_threshold'] = mh_ratio(cells, cv['assignment'](m - 1)[0])

    # single-item greedy order (ties at the last needed gain)
    gains = sorted(((h[i] if narrow0[i] else -h[i], i) for i in range(N)), key=lambda x: (-x[0], x[1]))
    acc, used, boundary = 0, 0, None
    for g, i in gains:
        if g <= 0:
            break
        acc += g
        used += 1
        if acc >= D0:
            boundary = g
            break
    out['greedy_changes'] = used
    out['greedy_items_tied_at_last_gain'] = sum(1 for g, _ in gains if g == boundary)

    # non-differential relabeling grid
    points = invalid = 0
    max_dev, region_ok = 0.0, True
    for ip in range(101):
        for iq in range(101):
            points += 1
            p, q = ip / 100, iq / 100
            if (ip, iq) in ((100, 0), (0, 100)):
                invalid += 1
                continue
            num = den = diff = 0.0
            for nE, eE, nN, eN in tables0:
                n = nE + nN
                xN, xE = (1 - q) * nN + p * nE, q * nN + (1 - p) * nE
                a, c = (1 - q) * eN + p * eE, q * eN + (1 - p) * eE
                if xE > 0 and xN > 0:
                    num += a * xE / n
                    den += c * xN / n
                diff += a - (eE + eN) * xN / n
            max_dev = max(max_dev, abs(diff - (1 - p - q) * D0 / L))
            at_or_below = num - den <= 1e-9 * max(1.0, den)
            if ip + iq != 100 and (ip + iq > 100) != at_or_below:
                region_ok = False
    out['grid_points'] = points
    out['grid_step'] = 0.01
    out['grid_points_with_empty_group'] = invalid
    out['grid_identity_holds'] = int(max_dev < 1e-8)
    out['grid_ratio_above_1_iff_p_plus_q_below_1'] = int(region_ok)

    # linked items with three independent labels
    link = read_csv(data_dir, 'panel_linked_labels.csv')
    out['linked_items'] = len(link)
    for b in BENCHMARKS:
        out['linked_items_' + b] = sum(1 for r in link if items[index[r['item_id']]]['benchmark'] == b)
    agreed = [r for r in link if r['unanimous_label']]
    out['linked_items_agreed'] = len(agreed)
    out['linked_items_agreed_DisTEMIST'] = sum(1 for r in agreed if items[index[r['item_id']]]['benchmark'] == 'DisTEMIST')
    out['linked_items_agreed_DisTEMIST_narrow'] = sum(1 for r in agreed if items[index[r['item_id']]]['benchmark'] == 'DisTEMIST'
                                                      and r['unanimous_label'] == 'NARROW')
    med = [r for r in agreed if items[index[r['item_id']]]['benchmark'] == 'MedProcNER']
    med_idx = {index[r['item_id']] for r in med}
    agreed_label = {index[r['item_id']]: r['unanimous_label'] for r in med}
    med_cells = [c for c in cells if c['benchmark'] == 'MedProcNER']
    out['linked_MedProcNER_agreed'] = len(med)

    def subset(keep, label_of, prefix):
        tables = []
        for c in med_cells:
            nE = eE = nN = eN = 0
            for i, e in zip(c['idx'], c['err']):
                if i not in keep:
                    continue
                if label_of(i) == 'NARROW':
                    nN += 1
                    eN += e
                else:
                    nE += 1
                    eE += e
            tables.append((nE, eE, nN, eN))
            k = prefix + '_' + c['system']
            out[k + '_exact_items'], out[k + '_exact_errors'], out[k + '_narrow_items'], out[k + '_narrow_errors'] = nE, eE, nN, eN
            out[k + '_risk_ratio'] = (eN / nN) / (eE / nE) if nE and nN and eE else None
        out[prefix + '_risk_ratio'] = mh_ratio_from_tables(tables)[0]

    subset(med_idx, lambda i: items[i]['label'], 'linked_agreed_panel_labels')
    subset(med_idx, lambda i: agreed_label[i], 'linked_agreed_agreed_labels')
    subset({i for i in range(N) if items[i]['benchmark'] == 'MedProcNER'}, lambda i: items[i]['label'], 'MedProcNER_all')
    relabeled = list(narrow0)
    for r in agreed:
        relabeled[index[r['item_id']]] = 1 if r['unanimous_label'] == 'NARROW' else 0
    out['full_panel_agreed_linked_labels_risk_ratio'] = mh_ratio(cells, relabeled)
    return out
