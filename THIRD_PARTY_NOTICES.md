# Third-party notices

The resources below are not distributed with this package and remain under their own terms. The license of this
package (see `LICENSE`) covers only the material created by its authors and grants no rights in these resources.

## SNOMED CT

SNOMED CT is a registered trademark of SNOMED International. The International Edition and the Spanish Edition release
files are distributed under the SNOMED CT license of SNOMED International and its member national release centres.
This package contains no release file and no SNOMED CT identifier, term, fully specified name, description or
relationship. The tables in `data/terminology_release/` are counts and aggregate statistics derived from release files.
The scripts in `licensed_inputs/` read release files that users must obtain under their own license; the identifiers
of the SNOMED CT metadata concepts that they need are supplied by the user as command-line arguments and are not
written in the package.

## Benchmark corpora

This package contains per-item outcomes, scores, relation types, granularity labels and document groupings derived from
system output on the three corpora below. It contains no corpus text, no mention and no corpus document identifier;
documents are represented by neutral sequential labels. The three corpora are released under the Creative Commons
Attribution 4.0 International license (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).

- DisTEMIST: Miranda-Escalada A, Farré E, Gasco L, Lima S, Krallinger M. DisTEMIST corpus: detection and normalization
  of disease mentions in Spanish clinical cases. Version 5.1. Zenodo; 2022. https://doi.org/10.5281/zenodo.7614764
- MedProcNER: Lima López S, Farré Maduell E, Gascó Sánchez L, Krallinger M. MedProcNER corpus: gold standard
  annotations for clinical procedures information extraction. Zenodo; 2023. https://doi.org/10.5281/zenodo.8224056
- SympTEMIST: Lima López S, Gascó Sánchez L, Farré E, Vigil Gimenez L, Krallinger M. SympTEMIST corpus: gold standard
  annotations for clinical symptoms, signs and findings information extraction. Version 4. Zenodo; 2024.
  https://doi.org/10.5281/zenodo.10635215

The Spanish lexical retrieval (`licensed_inputs/spanish_lexical_retrieval.py`) reads the organizers' public
terminology files: the DisTEMIST gazetteer, released separately from the corpus, and the gazetteers included in the
MedProcNER and SympTEMIST releases above. These files contain SNOMED CT identifiers and terms and are not distributed
with this package; it contains only per-mention outcomes and counts from that step. The DisTEMIST gazetteer is
released under CC BY 4.0:

- DISTEMIST gazetteer: Miranda-Escalada A, Gascó L, Krallinger M. DISTEMIST gazetteer. Version 2.0. Zenodo; 2022.
  https://doi.org/10.5281/zenodo.6505583

MedMentions enters only as aggregate counts and means by depth band (`data/medmentions_bands.csv`): Mohan S, Li D.
MedMentions: a large biomedical corpus annotated with UMLS concepts. Automated Knowledge Base Construction (AKBC);
2019. https://doi.org/10.24432/C5G59C. The corpus is released under CC0 1.0
(https://creativecommons.org/publicdomain/zero/1.0/).

## Thai term-mapping table

The Thai term-level evaluation uses the interface terminology mapping table published as Multimedia Appendix 2 of
Tangchitnob N, Ponthongmak W, Kijsanayotin B, Pattanaprateep O, Phusanti S, Atiksawedparit P, Suwanthaweemeesuk K,
Siangfu J, McKay GJ, Attia J, Thakkinstian A. Developing a Thai user interface terminology for Systematized
Nomenclature of Medicine Clinical Terms implementation in primary care: cross-sectional content coverage analysis.
JMIR Med Inform. 2026;14:e80039. https://doi.org/10.2196/80039. The article is an open-access article distributed
under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). This package contains only per-term errors and
per-cluster counts with neutral positions; it contains no Thai or English term, no candidate label and no concept
identifier from that table.

## Cross-check systems

`data/xmen_strata.csv` holds aggregate results of the xMEN toolkit (Apache-2.0; Borchert F, Llorca I, Roller R,
Arnrich B, Schapranow MP. xMEN: a modular toolkit for cross-lingual medical entity normalization. JAMIA Open.
2025;8(1):ooae147. https://doi.org/10.1093/jamiaopen/ooae147) run by the authors, and `data/participant_runs.csv`
aggregate rescoring results of five publicly released shared-task runs (Almeida T, Jonker RAA, Poudel R, Silva JM,
Matos S. BIT.UA at MedProcNer: discovering medical procedures in Spanish using transformer models with MCRF and
augmentation. CEUR Workshop Proceedings, vol. 3497; 2023. p. 60-72. https://ceur-ws.org/Vol-3497/paper-005.pdf; runs
released at https://github.com/ieeta-pt/MedProcNER). Neither file contains predictions, mention text or concept
identifiers.

## Models

The per-item outcomes were derived from the outputs of the following models, which are not distributed with this
package:

- Qwen2.5-32B-Instruct (https://huggingface.co/Qwen/Qwen2.5-32B-Instruct): Apache-2.0
- SapBERT-from-PubMedBERT-fulltext (https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext): Apache-2.0
- NLLB-200, nllb-200-distilled-600M (https://huggingface.co/facebook/nllb-200-distilled-600M): CC BY-NC 4.0

The package contains no model weights and no model output text (translations, generated text, rankings or prompts);
it contains only 0/1 indicators and scores computed from those outputs.

## Runtime

Python 3.9 or later with its standard library. No third-party package is bundled or required. `analysis/pcg64.py`
reimplements in pure Python the PCG64 generator, SeedSequence seeding, bounded integer and multinomial sampling of
NumPy (BSD-3-Clause license) to reproduce its random streams; it contains no NumPy code. Algorithm sources:

- O'Neill ME. PCG: a family of simple fast space-efficient statistically good algorithms for random number
  generation. Technical report HMC-CS-2014-0905. Claremont (CA): Harvey Mudd College; 2014.
  https://www.pcg-random.org/paper.html
- Lemire D. Fast random integer generation in an interval. ACM Trans Model Comput Simul. 2019;29(1):3.
  https://doi.org/10.1145/3230636
- NumPy documentation: Permuted Congruential Generator (64-bit, PCG64),
  https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html; SeedSequence,
  https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html; Random
  Generator, https://numpy.org/doc/stable/reference/random/generator.html
