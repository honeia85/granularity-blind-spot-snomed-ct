# Data dictionary

All files are UTF-8 CSV with a header row and LF line endings. Labels: EXACT means the mention is at the same level
as its assigned concept; NARROW means the mention is more specific than its assigned concept (as released with the
corpus unless stated otherwise). Benchmarks: DisTEMIST, MedProcNER, SympTEMIST.

## Fixed panel

`data/panel_items.csv` - 2,103 mentions of the fixed panel (678 DisTEMIST, 625 MedProcNER, 800 SympTEMIST), one row
each. The row order is the processing order that decides ties in the label-budget search.
- `item_id`: neutral item label (P0001-P2103)
- `benchmark`
- `label`: EXACT or NARROW, as released with the corpus
- `document_id`: neutral document label (D001-D245); the same label in two benchmarks denotes the same source
  document. The joint document resampling of the main analysis indexes documents in label order
- `position`: position of the mention within its benchmark (1 to the benchmark size); the main analysis orders the
  mentions of a benchmark by this column, which fixes the mention resampling and the bootstrap draws

`data/panel_documents.csv` - the 245 source documents, one row each.
- `document_id`: as in `panel_items.csv`
- `validation_position`: index of the document (0-244) in the document resampling of the label validation

`data/mention_scores.csv` - outcome of every fixed-panel mention for each system in both candidate spaces (19,780 rows:
14 cells x 2 spaces).
- `item_id`: as in `panel_items.csv`
- `candidate_space`: `oracle` (gold-inclusive inventory: training gold codes plus the gold codes of the
  evaluated test items) or `train` (training-only inventory)
- `system`: `lexical`, `sapbert`, `sapbert_ft` (fine-tuned SapBERT), `llm_rerank` (language-model reranking) or
  `llm_gen` (language-model generation); MedProcNER has no `sapbert_ft` cell
- `exact`: 1 if the top-ranked prediction equals the gold code, else 0
- `is_a`: 1 if the prediction is the gold code, an ancestor or a descendant of it, else 0
- `wup`: Wu-Palmer score (0 to 1, at most four decimals; 0 for off-map pairs)
- `relation_type`: `exact`, `ancestor` (the prediction is an ancestor of the gold code), `descendant`, `sibling`
  (sibling or cousin) or `unresolved`

`data/panel_linked_labels.csv` - 68 panel items with an exact link to mentions that carry three independent labels.
- `item_id`: as in `panel_items.csv`
- `unanimous_label`: EXACT or NARROW when all three labels agree, empty otherwise
- `majority_label`: label given by at least two of the three labels

## Re-annotation by three annotators

`data/reannotation_labels.csv` - balanced sample of 150 mentions (25 corpus-EXACT and 25 corpus-NARROW per benchmark),
one row each, in label order.
- `item_id`: neutral label (A001-A150)
- `benchmark`
- `corpus_label`: EXACT or NARROW, as released with the corpus
- `R1`, `R2`, `R3`: label assigned by each annotator (EXACT or NARROW)

## Cross-checks

`data/xmen_strata.csv` - xMEN (SapBERT candidate generation) on separate stratified 800-mention draws, by benchmark,
input and native label.
- `benchmark`; `input`: `translated` (mentions translated to English) or `untranslated`
- `label`: EXACT or NARROW
- `n`: mentions; `exact_count`: exact matches; `wup_mean`: mean Wu-Palmer score

`data/medmentions_bands.csv` - an 800-mention English MedMentions sample by depth band of the gold concept, for the
lexical and SapBERT systems and for xMEN.
- `system`: `lexical`, `sapbert` or `xmen`
- `depth_band`: `shallow` (depth <= 2), `mid` (3) or `deep` (>= 4)
- `n`; `exact_count`; `is_a_count`: is-a consistent predictions; `wup_mean`

`data/participant_runs.csv` - five public shared-task runs on MedProcNER, rescored by native label.
- `run`: 1-5
- `domain`: `all_matched` (all matched predictions, frozen graph), `graph_resolvable` (predictions resolvable in the
  frozen graph), `all_matched_reference_release` (all matched predictions with numeric identifiers, SNOMED CT
  hierarchy of 17 October 2025) or `resolvable_reference_release` (the resolvable subset of the latter)
- `n_exact`, `n_narrow`: predictions by label
- `exact_mean_exact`, `wup_mean_exact`, `exact_mean_narrow`, `wup_mean_narrow`: mean exact match and Wu-Palmer score
  by label (four decimals in the first two domains)

## Label validation

`data/label_validation_judgments.csv` - 300 panel mentions (50 per benchmark and corpus label), one row per mention
and clinician (600 rows).
- `item_id`: neutral label (LV001-LV300)
- `benchmark`
- `corpus_label`: EXACT or NARROW, as released with the corpus
- `rater`: A or B
- `choice`: `same` (same level as the assigned concept), `narrower` (more specific than the assigned concept) or `hold`

## Alternative codes

`data/alternative_code_judgments.csv` - 100 NARROW mentions, stratified by benchmark (41 DisTEMIST, 24 MedProcNER,
35 SympTEMIST), one row each.
- `item_id`: neutral label (A001-A100)
- `benchmark`
- `rater_A_decision`, `rater_B_decision`: `selected` (at least one presented candidate is an acceptable alternative),
  `none` or `hold`
- `rater_A_candidates`, `rater_B_candidates`: selected candidates as neutral labels separated by `;` (`c1`, `c2`, ... ,
  numbered within each mention); empty unless the decision is `selected`
- `consensus_decision`, `consensus_candidates`: consensus judgment, given only for the mentions on which the
  judgments of raters A and B disagree

## Thai term-level evaluation

`data/thai_clusters.csv` - counts per reference-concept cluster, for each system.
- `system`: `lexical`, `sapbert` or `sapbert_tag_removed` (candidate labels without the trailing parenthetical tag)
- `position`: order of the cluster in the resampling (1-852)
- `exact_items`, `exact_errors`, `narrow_items`, `narrow_errors`: items and errors of the cluster by label

`data/thai_items.csv` - one row per term (999 per system). The terms are the single-row ACCEPTED entries of the
mapping table with relationship type TARGET_EQUIVALENT or TARGET_BROADER, a Term value in Latin letters and a target
code that passes the code checks described in `README.md` (Thai term-level evaluation); the candidates are the 1,421
target codes of the ACCEPTED rows that pass the same checks.
- `system`: as above
- `position`: order of the item in the resampling (1-999)
- `label`: EXACT or NARROW (relation of the term to its reference concept in the mapping table)
- `error`: 1 if the top-ranked candidate is not the reference identifier, else 0

## Translation-by-normalization control

`data/translation_cases.csv` - 2,336 mentions, one row each.
- `document`: document group number (1-246); groups are resampled in this order
- `benchmark`, `label`
- for each arm `nllb_sapbert`, `nllb_rerank`, `qwen_sapbert`, `qwen_rerank`, `nllb_generator`, `qwen_generator`
  (translation model, then normalization method):
  - `<arm>_error`: 1 if the output could not be parsed or its code differs from the reference code, else 0
  - `<arm>_parse_failure`: 1 if the output could not be parsed, else 0

## Spanish lexical retrieval

`data/spanish_lexical_mentions.csv` - one row per fixed-panel mention (2,103).
- `item_id`: neutral label (SL0001-SL2103)
- `benchmark`, `label`
- `gazetteer_exact`, `spanish_edition_exact`: 1 if the top-ranked concept equals the reference code with gazetteer
  names or with Spanish-edition synonyms (Spanish Edition release of 10 August 2026), else 0

`data/spanish_lexical_counts.csv` - counts by benchmark, as written by `licensed_inputs/spanish_lexical_retrieval.py`.
- `benchmark`; `candidate_codes`: candidate concepts named in both resources
- `n_EXACT`, `n_NARROW`: mentions by label
- `gazetteer_exact_EXACT`, `gazetteer_exact_NARROW`, `spanish_edition_exact_EXACT`, `spanish_edition_exact_NARROW`:
  exact matches by resource and label

## Terminology release

Release dates are written as YYYY-MM-DD. The tables were computed from the International Edition release of 1 November
2026 (Full files) and the Spanish Edition release of 31 October 2022. The reference time point is the International
release of 17 October 2025.

`data/terminology_release/time_points.csv` - one row per time point (seven International releases and the Spanish
edition).
- `time_point`: `international_<date>` or `spanish_edition`
- `international_release`: International release used (for the Spanish edition, the release it declares as its base)
- `inferred_edges`, `stated_edges`: is-a edges within the ancestor closure of the evaluated concepts
- `inferred_edges_added_vs_reference`, `inferred_edges_deleted_vs_reference`, `stated_edges_added_vs_reference`,
  `stated_edges_deleted_vs_reference`: edges added or deleted in moving from the time point to the reference
- `gold_active`, `gold_inactive`, `gold_absent`: states of the 1,517 gold concepts
- `rows_changed_vs_reference`, `rows_up`, `rows_down`: prediction rows (of 9,890) whose Wu-Palmer score, at four
  decimals, differs from the reference, in total and by direction
- `cells_with_changed_rows`: cells (of 14) with at least one changed row
- `credit_gap_ratio_geometric_mean`: geometric mean over the 14 cells of the NARROW/EXACT partial-credit-gap ratio

`data/terminology_release/cells.csv` - one row per International time point and cell.
- `time_point`, `benchmark`, `system`
- `exact_rows`, `narrow_rows`: rows by label
- `credit_gap_exact`, `credit_gap_narrow`: mean partial-credit gap (Wu-Palmer score minus exact match) by label
- `rows_changed_vs_reference`

`data/terminology_release/gold_states_by_benchmark.csv` - gold-concept states per International time point and
benchmark: `time_point`, `benchmark`, `gold_concepts`, `active`, `inactive`, `absent`.

`data/terminology_release/compatible_releases.csv` - International releases (distinct effective times of the concept
file) compatible with the gold concepts of each benchmark.
- `benchmark`, `gold_concepts`
- `first_release_all_exist`: first release in which every gold concept exists
- `first_release_all_active`, `last_release_all_active`, `releases_all_exist_and_active`: releases in which every gold
  concept exists and is active (empty when there is none)
- `gold_inactive_at_20221031`: gold concepts inactive in the release of 31 October 2022

`data/terminology_release/summary.csv` - `quantity`, `value`: counts of evaluated, gold and predicted concepts and of
prediction rows, the reference release, the Spanish release and its declared International base, and the content of
the Spanish extension (concepts, is-a rows, active and inactive is-a rows, extension concepts among the evaluated
concepts, prediction rows whose score changes when the extension is added to its base).

`data/terminology_release/edition_statements.csv` - SNOMED CT editions stated in the organizer documentation:
`benchmark`, `statement`, `date` (YYYY-MM-DD; empty when not stated), `source`.

## Results and expected values

`results/<analysis>.json` - named quantities computed by `analysis/<analysis>.py` (keys sorted; null where a quantity
is undefined). `run_all.py` writes the same files to `output/`.

`expected/reported_values.csv` - values printed in the article and its additional file (1,258 rows).
- `analysis`, `quantity`: key of a quantity in `results/<analysis>.json`
- `reported`: value as printed (thousands separators removed; dates as YYYY-MM-DD)
- `scale`: multiplier applied before rounding (100 for percentages, -1 for deleted edges printed with a minus sign,
  1000 for a P value printed as a multiple of 10^-3)
