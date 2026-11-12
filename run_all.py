#!/usr/bin/env python3
"""Recompute all results of the package and compare them with the values reported in the article.

Usage: python3 run_all.py [--out DIR]

Each analysis in analysis/ reads the files in data/ and writes one JSON file of named quantities to DIR
(default: output). Every row of expected/reported_values.csv names a quantity and the value printed in the article;
the computed quantity, multiplied by the given scale (100 for percentages, -1 for counts of deleted relationships
printed with a minus sign; 1000 for a P value printed as a multiple of 10^-3), rounded half up to the number of
decimals of the printed value, must equal the printed value; dates (YYYY-MM-DD) must be equal as written. Rounding starts from the exact binary value or from its
shortest decimal representation; the two differ only for values that lie on a rounding boundary. The computed files
are also compared byte for byte with the files in results/. Python 3.9 or later, standard library only.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
sys.path.insert(0, os.path.join(HERE, 'analysis'))

import alternative_codes  # noqa: E402
import annotator_agreement  # noqa: E402
import external_checks  # noqa: E402
import label_sensitivity  # noqa: E402
import label_validation  # noqa: E402
import main_intervals  # noqa: E402
import main_scores  # noqa: E402
import spanish_lexical  # noqa: E402
import terminology_release  # noqa: E402
import thai_terms  # noqa: E402
import translation_control  # noqa: E402

ANALYSES = [
    ('main_scores', main_scores),
    ('main_intervals', main_intervals),
    ('annotator_agreement', annotator_agreement),
    ('external_checks', external_checks),
    ('alternative_codes', alternative_codes),
    ('label_sensitivity', label_sensitivity),
    ('label_validation', label_validation),
    ('thai_terms', thai_terms),
    ('translation_control', translation_control),
    ('spanish_lexical', spanish_lexical),
    ('terminology_release', terminology_release),
]


def matches(value, reported, scale):
    if value is None or isinstance(value, bool):
        return False
    if DATE.match(reported):
        return value == reported
    target = Decimal(reported)
    exponent = target.as_tuple().exponent
    unit = Decimal(1).scaleb(exponent)
    for exact in (Decimal(value), Decimal(repr(value))):
        if (exact * Decimal(scale)).quantize(unit, rounding=ROUND_HALF_UP) == target:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'output'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data_dir = os.path.join(HERE, 'data')
    results = {}
    for name, module in ANALYSES:
        start = time.time()
        res = module.run(data_dir)
        results[name] = res
        text = json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + '\n'
        with open(os.path.join(args.out, name + '.json'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        shipped = os.path.join(HERE, 'results', name + '.json')
        same = os.path.exists(shipped) and open(shipped, encoding='utf-8').read() == text
        print('%-20s computed in %5.1f s; identical to results/%s.json: %s' % (name, time.time() - start, name, 'yes' if same else 'NO'))

    with open(os.path.join(HERE, 'expected', 'reported_values.csv'), encoding='utf-8', newline='') as f:
        expected = list(csv.DictReader(f))
    failures = []
    counts = {}
    for row in expected:
        value = results[row['analysis']].get(row['quantity'])
        ok = matches(value, row['reported'], row['scale'])
        counts.setdefault(row['analysis'], [0, 0])[0 if ok else 1] += 1
        if not ok:
            failures.append((row['analysis'], row['quantity'], row['reported'], value))
    print()
    for name, _ in ANALYSES:
        p, f = counts.get(name, [0, 0])
        print('%-20s reported values: %3d pass, %d fail' % (name, p, f))
    for a, q, r, v in failures:
        print('FAIL %s.%s: reported %s, computed %r' % (a, q, r, v))
    print()
    print('ALL %d REPORTED VALUES PASS' % len(expected) if not failures else '%d OF %d REPORTED VALUES FAIL' % (len(failures), len(expected)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
