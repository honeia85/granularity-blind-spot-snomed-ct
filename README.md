# Reproduction package: A granularity blind spot in the evaluation of SNOMED CT concept normalization

Data, code and results for the reported analyses that do not require licensed or identifier-linked inputs, from a
study of mention-level evaluation of SNOMED CT concept normalization on three Spanish clinical benchmarks (DisTEMIST,
MedProcNER, SympTEMIST), with cross-checks on MedMentions, the xMEN toolkit, public shared-task runs and a Thai
term-mapping table. Reported values that need such inputs are listed under "Values that require licensed or
non-distributed inputs".

## Main analysis

| Analysis | Code | Data | Results |
|---|---|---|---|
| Scores of five systems in 14 system x benchmark cells and two candidate spaces; credit gaps (Wu-Palmer minus exact match) by native granularity label; error incidence, conditional proximity and their decomposition; Mantel-Haenszel risk and odds ratios; Breslow-Day tests; binary is-a uplift; relation types of non-exact predictions; fine-tuning gains and McNemar tests; sample sizes | `analysis/main_scores.py` | `panel_items.csv`, `mention_scores.csv` | `results/main_scores.json` |
| Intervals: joint resampling of the 245 source documents (primary interval), mention resampling within benchmarks, credit-gap ratio bootstrap, additive credit-gap bootstrap in both candidate spaces, fixed-stratum item bootstrap for conditional proximity and is-a uplift | `analysis/main_intervals.py` | as above | `results/main_intervals.json` |
| Re-annotation of the granularity label by three clinical annotators (Cohen's and Fleiss' kappa, majority vote) | `analysis/annotator_agreement.py` | `reannotation_labels.csv` | `results/annotator_agreement.json` |
| Cross-checks from aggregate tables: xMEN on separate 800-mention draws, MedMentions depth bands, five public shared-task runs on MedProcNER | `analysis/external_checks.py` | `xmen_strata.csv`, `medmentions_bands.csv`, `participant_runs.csv` | `results/external_checks.json` |

## Supplementary analyses

| Analysis | Code | Data | Results |
|---|---|---|---|
| Sensitivity of the NARROW/EXACT error risk ratio to the granularity labels (exact label budgets, non-differential relabeling, items with three independent labels) | `analysis/label_sensitivity.py` | `panel_items.csv`, `mention_scores.csv`, `panel_linked_labels.csv` | `results/label_sensitivity.json` |
| Label validation by two clinicians (agreement, misclassification probabilities, corrected risk ratio) | `analysis/label_validation.py` | `label_validation_judgments.csv`, `panel_documents.csv` and the panel files | `results/label_validation.json` |
| Alternative acceptable codes for NARROW mentions (agreement of raters A and B, design-weighted proportion) | `analysis/alternative_codes.py` | `alternative_code_judgments.csv`, `panel_items.csv` | `results/alternative_codes.json` |
| Hierarchical relation of the accepted alternative codes to the reference concepts, and the NARROW exact-match penalty when predictions that match an accepted alternative are credited (licensed inputs; see below) | `licensed_inputs/alternative_code_hierarchy.py` | identifier-linked review file (not included), `panel_items.csv`, `mention_scores.csv`, `alternative_code_judgments.csv` | `results/alternative_code_hierarchy.json` |
| Thai term-level evaluation (lexical baseline and SapBERT retrieval, cluster and item bootstrap) | `analysis/thai_terms.py` | `thai_clusters.csv`, `thai_items.csv` | `results/thai_terms.json` |
| Translation-by-normalization control (NLLB or Qwen translation, SapBERT retrieval or Qwen reranking) | `analysis/translation_control.py` | `translation_cases.csv` | `results/translation_control.json` |
| Spanish lexical retrieval with a public gazetteer and with Spanish-edition synonyms | `analysis/spanish_lexical.py` | `spanish_lexical_mentions.csv`, `spanish_lexical_counts.csv` | `results/spanish_lexical.json` |
| Terminology release (concept states, is-a closure and partial credit at seven International release dates and the Spanish edition) | `analysis/terminology_release.py` | `terminology_release/` | `results/terminology_release.json` |

All data files are in `data/`. `DATA_DICTIONARY.md` describes every file and column; the method of each analysis is
described at the top of its code file.

## Requirements

Python 3.9 or later. Only the standard library is used; no package needs to be installed.

## Run

From the package root:

```
python3 run_all.py
```

`run_all.py` recomputes every result file of `results/` from `data/` (except `results/alternative_code_hierarchy.json`, which needs licensed inputs), writes the results to `output/`, reports
whether each file is identical to the corresponding file in `results/`, and compares 1,258 values with the values
printed in the article and its additional file (`expected/reported_values.csv`). It ends with
`ALL 1258 REPORTED VALUES PASS` or lists the values that differ. A full run takes about two minutes. File integrity can be checked with
`shasum -a 256 -c SHA256SUMS` (or `sha256sum -c SHA256SUMS`).

## Seeds and random streams

Python's `random.Random`:
- 12345: joint resampling of the source documents and mention resampling within benchmarks (main analysis); document
  bootstrap of the translation control.
- 20260805: bootstrap of the NARROW/EXACT credit-gap ratio and of the additive credit gap.
- 20261111: label validation.

PCG64 generator of numpy, reimplemented in `analysis/pcg64.py` so that numpy is not needed (the draws are identical to
those of `numpy.random.default_rng(seed)`):
- 42: fixed-stratum item bootstrap (conditional proximity, is-a uplift) and the re-annotation kappa intervals
  (`integers`).
- 12345: cluster bootstrap of the Thai evaluation (`multinomial`).
- 54321: item bootstrap of the Thai evaluation (`multinomial`).

Linear congruential generator described in `licensed_inputs/alternative_code_hierarchy.py`:
- 20261112: resampling of the reviewed mentions in the alternative-code hierarchy analysis.

## Analyses that need licensed or non-distributed inputs

- Terminology release: the tables in `data/terminology_release/` were computed by
  `licensed_inputs/terminology_release_tables.py` from the SNOMED CT International Edition release of 1 November 2026
  (its Full files cover all seven time points), the Spanish Edition release of 31 October 2022 and the predictions of
  the 14 gold-inclusive cells (which contain SNOMED CT identifiers). Users with a SNOMED CT license can regenerate them;
  `run_all.py` checks their internal consistency.
- Spanish lexical retrieval: `licensed_inputs/spanish_lexical_retrieval.py` produces the two Spanish lexical files from
  the mention texts, the organizers' public terminology files and the Spanish Edition release of 10 August 2026. The numbers of candidate codes
  named in both resources come from that step; all other counts are recomputed by `run_all.py` from the mention-level
  outcomes.
- Alternative-code hierarchy: `licensed_inputs/alternative_code_hierarchy.py` computes `results/alternative_code_hierarchy.json`
  from the SNOMED CT International Edition release of 1 November 2026 (hierarchies of 17 October 2025 and 1 November
  2026 from its Full files) and a review file with the reference code, the presented and accepted candidate codes and
  the top-ranked predictions of the five systems for each of the 100 reviewed mentions (all SNOMED CT identifiers, so
  the file is not included). The NARROW and EXACT error counts of the 14 gold-inclusive cells are read from `data/`. The
  result file holds aggregate counts, proportions and risk ratios only.
- The three scripts in `licensed_inputs/` take the identifiers of the SNOMED CT metadata concepts they need (is-a
  relationship type, inferred characteristic type, International core module, synonym description type) as required
  command-line arguments. These identifiers are not written in the package; users look them up in the documentation
  of the SNOMED CT release they hold and pass them when running the scripts.
- Per-mention scores: exact match, is-a consistency, Wu-Palmer score and relation type were computed by scoring the
  system predictions against the gold codes on a frozen SNOMED CT is-a graph; the package contains the resulting
  scores. Recall@5 needs the ranked candidate lists and is not included.
- Thai term-level evaluation: the 999 terms are the source codes of the mapping table that have a single row with
  status ACCEPTED, relationship type TARGET_EQUIVALENT (EXACT, 952 terms) or TARGET_BROADER (NARROW, 47 terms), a
  value in the Term column written in Latin letters without Thai characters, and a target code that passes the SNOMED
  CT check digit (Verhoeff) and is not given as a numeric cell of more than 15 digits, which cannot hold every digit
  of the code. The 1,421 candidate identifiers are the distinct target codes of all rows with status ACCEPTED that
  pass the same code checks, across all relationship types. The SapBERT predictions (first-token embeddings, maximum
  input length 32, cosine similarity, top-ranked candidate) and the lexical baseline were computed on the
  third-party table; the package contains the resulting errors.
- Cross-checks: the xMEN, MedMentions and shared-task tables are aggregate counts and means by stratum; the package
  recomputes the derived quantities (overall means, credit gaps, ratios, resolvable shares) from them.

## Values that require licensed or non-distributed inputs

The values below are reported in the article or its additional file and are computed in steps that need licensed
SNOMED CT content, SNOMED CT identifiers, corpus text or other non-distributed inputs.

- Recall@5 (Table 2): needs the ranked candidate lists, which contain SNOMED CT identifiers.
- MedMentions LLM reranker (its row of Table 2 and its depth-band credit gaps in Additional file 1): needs its
  per-mention outputs, which contain identifiers.
- Covariate-adjusted modified-Poisson model on SympTEMIST (risk ratio 1.77 before and 1.55 after adjustment, and the
  covariate risk ratios): the covariates need mention text and identifier-linked concept data (training frequency,
  candidate-space density, depth).
- Seen/novel split of the full DisTEMIST test set (n = 2,507: code-seen, ancestor-seen and fully isolated shares,
  distance to the nearest training ancestor, and the nearest-ancestor ceiling of mean Wu-Palmer 0.837): needs the gold
  and training identifiers and the is-a graph.
- Corpus-level counts (training rows, unique codes and eligible test rows): need the corpus annotation files, which
  contain SNOMED CT identifiers.
- Depth-proxy validity (odds ratios per depth level and AUC): needs the concept depths of the Spanish items, which are
  identifier-linked.
- Fine-tuning exposure buckets (accuracies of code-seen, ancestor-seen and fully isolated mentions and their
  differences): needs the codes of the fine-tuning pairs.
- Descendant-availability restriction (Supplementary Table S14 and the related text: the share of mentions whose gold
  concept has at least one descendant in the frozen graph, the error rates and NARROW/EXACT incidence ratios after
  restriction, and the restricted Mantel-Haenszel risk ratio): needs the frozen ontology map, which is licensed SNOMED
  CT content.
- Sizes of the candidate spaces, the ancestor universes and the xMEN concept space, and the numbers of codes removed in
  the training-only sensitivity: counted from identifier lists.
- Composition of the initial draws before de-duplication (EXACT/NARROW rows per benchmark): needs the corpus rows.
- Reranker fallback counts and rates: need the raw reranker outputs, which contain identifiers.
- Share of non-exact predictions that share only the SNOMED CT root: needs identifiers and the is-a graph.
- Alternative metrics (normalized inverse LCA path distance, ancestor-set Jaccard) and their rank agreement with
  Wu-Palmer: need the frozen is-a graph.
- Agreement of the NLLB and Qwen translation strings and the corresponding SapBERT gap ratios (Additional file 1,
  Extended limitations): need the translated mention text.
- Candidate retrieval for the alternative-code review (minimum word length, maximum candidates per mention, numbers of
  candidates and of mentions with and without a candidate): the candidate lists contain identifiers.
- Alignment of the public shared-task runs with the corpus (share of matched predictions, surface-text agreement,
  agreement of the gaps with the organizers' scoring): needs the participants' prediction files and the corpus text.
- xMEN tokenizer check (token-identifier mismatches): needs the mention text.
- Agreement of the hierarchy of 17 October 2025 with the primary credit gaps in all 14 cells: needs licensed release
  files.
- Hierarchical relation of the accepted alternative codes and the risk ratios after crediting matching predictions
  (Additional file 1, alternative-code review; `results/alternative_code_hierarchy.json`): need the reference,
  candidate and predicted codes of the reviewed mentions and licensed release files.
- Values quoted from cited publications (for example, the agreement reported for the Thai mapping table): taken from
  the source articles.

## What is not included

No SNOMED CT identifier, term, description, relationship or release file; no corpus text (mention text or clinical
context); no translation, generated text, prompt or model weights; and no Thai or English term of the mapping table.
Item, document, cluster and candidate labels in the data files
are neutral sequential labels (for example `P0001`, `D001`, `c1`) that carry no information beyond grouping.

## License

Code (`.py` files): MIT License. Data, results and documentation: CC BY 4.0. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`. Citation metadata: `CITATION.cff`.
