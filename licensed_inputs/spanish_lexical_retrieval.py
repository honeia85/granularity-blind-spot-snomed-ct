#!/usr/bin/env python3
"""Spanish lexical retrieval with two vocabularies: a public gazetteer and Spanish-edition synonyms.

For each benchmark, every mention is matched without translation against the candidate concepts that have
names in both vocabularies. Each concept is represented by its names joined into one string; mention and
concept strings are compared by character n-grams (lengths 2 to 4, lower-cased, whitespace collapsed,
padded with one space) weighted by sublinear term frequency, 1 + ln(tf), and inverse document frequency,
ln((N + 1) / (df + 0.5)), where N is the number of candidate concepts. The top-ranked concept by cosine
similarity is the prediction (the first concept reached wins ties); a prediction is an exact match when it
equals the gold code.

Inputs (all given as arguments; none is distributed with the package):
  --mentions    CSV with columns item_id, benchmark, label (EXACT/NARROW), mention, gold_code
  --candidates  directory with <benchmark>.txt, one candidate concept identifier per line
  --gazetteer   <benchmark>=<path> for each benchmark, a TSV with columns code and term
  --rf2         Spanish edition release archive (zip); active Spanish-language synonyms of the
                Snapshot description file are used (the reported values were obtained with the Spanish
                edition released on 10 August 2026)
  --synonym-type-id
                identifier of the SNOMED CT metadata concept for the synonym description type. It is not
                written in this file; take it from the documentation of the SNOMED CT release you use.
Outputs (in --out): spanish_lexical_mentions.csv and spanish_lexical_counts.csv, the two files in data/.
Standard library only.
"""
import argparse, collections, csv, io, math, os, zipfile


def grams(text):
    s = ' ' + ' '.join(text.lower().split()) + ' '
    out = []
    for n in (2, 3, 4):
        out.extend(s[i:i + n] for i in range(len(s) - n + 1))
    return out


class Retriever:
    def __init__(self, names_by_code):
        self.codes = [c for c in names_by_code if names_by_code[c]]
        counts = []
        df = collections.defaultdict(int)
        for c in self.codes:
            tf = collections.defaultdict(int)
            for g in grams(' ; '.join(sorted(names_by_code[c]))):
                tf[g] += 1
            counts.append(tf)
            for g in tf:
                df[g] += 1
        n = len(self.codes)
        self.idf = {g: math.log((n + 1) / (d + 0.5)) for g, d in df.items()}
        self.post = collections.defaultdict(list)
        self.norm = []
        for i, tf in enumerate(counts):
            ss = 0.0
            for g, k in tf.items():
                w = (1 + math.log(k)) * self.idf[g]
                if w:
                    self.post[g].append((i, w))
                    ss += w * w
            self.norm.append(math.sqrt(ss) or 1.0)

    def best(self, text):
        tf = collections.defaultdict(int)
        for g in grams(text):
            tf[g] += 1
        acc = collections.defaultdict(float)
        for g, k in tf.items():
            if g in self.idf:
                w = (1 + math.log(k)) * self.idf[g]
                for i, wi in self.post[g]:
                    acc[i] += w * wi
        top, top_s = None, -1.0
        for i, s in acc.items():
            s = s / self.norm[i]
            if s > top_s:
                top, top_s = i, s
        return self.codes[top if top is not None else 0]


def read_gazetteer(path):
    names = {}
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            code, term = (r.get('code') or '').strip(), (r.get('term') or '').strip()
            if code and term:
                names.setdefault(code, set()).add(term)
    return names


def read_rf2_synonyms(path, wanted, synonym_type):
    names = {}
    with zipfile.ZipFile(path) as z:
        member = next(m for m in z.namelist() if '/Snapshot/Terminology/sct2_Description_' in m)
        with z.open(member) as fb:
            rows = io.TextIOWrapper(fb, encoding='utf-8')
            next(rows)
            for line in rows:
                p = line.rstrip('\n').split('\t')
                if p[2] == '1' and p[4] in wanted and p[5] == 'es' and p[6] == synonym_type and p[7].strip():
                    names.setdefault(p[4], set()).add(p[7].strip())
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mentions', required=True)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--gazetteer', action='append', required=True, help='benchmark=path')
    ap.add_argument('--rf2', required=True)
    ap.add_argument('--synonym-type-id', required=True, help='identifier of the synonym description type')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    gaz_paths = dict(x.split('=', 1) for x in a.gazetteer)
    with open(a.mentions, encoding='utf-8', newline='') as f:
        mentions = list(csv.DictReader(f))
    benchmarks = list(dict.fromkeys(m['benchmark'] for m in mentions))
    cand = {}
    for b in benchmarks:
        with open(os.path.join(a.candidates, b + '.txt'), encoding='utf-8') as f:
            cand[b] = [l.strip() for l in f if l.strip()]
    rf2 = read_rf2_synonyms(a.rf2, set().union(*cand.values()), a.synonym_type_id)
    os.makedirs(a.out, exist_ok=True)
    outcomes, summary = [], []
    for b in benchmarks:
        gaz = read_gazetteer(gaz_paths[b])
        codes = [c for c in cand[b] if c in gaz and c in rf2]
        rows = [m for m in mentions if m['benchmark'] == b]
        hits = {}
        for name, vocab in (('gazetteer', gaz), ('spanish_edition', rf2)):
            r = Retriever({c: vocab[c] for c in codes})
            hits[name] = [int(r.best(m['mention']) == m['gold_code']) for m in rows]
        for i, m in enumerate(rows):
            outcomes.append([m['item_id'], b, m['label'], hits['gazetteer'][i], hits['spanish_edition'][i]])
        line = [b, len(codes)]
        for lab in ('EXACT', 'NARROW'):
            line.append(sum(1 for m in rows if m['label'] == lab))
        for name in ('gazetteer', 'spanish_edition'):
            for lab in ('EXACT', 'NARROW'):
                line.append(sum(h for h, m in zip(hits[name], rows) if m['label'] == lab))
        summary.append(line)
    with open(os.path.join(a.out, 'spanish_lexical_mentions.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['item_id', 'benchmark', 'label', 'gazetteer_exact', 'spanish_edition_exact'])
        w.writerows(outcomes)
    with open(os.path.join(a.out, 'spanish_lexical_counts.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['benchmark', 'candidate_codes', 'n_EXACT', 'n_NARROW', 'gazetteer_exact_EXACT', 'gazetteer_exact_NARROW',
                    'spanish_edition_exact_EXACT', 'spanish_edition_exact_NARROW'])
        w.writerows(summary)


if __name__ == '__main__':
    main()
