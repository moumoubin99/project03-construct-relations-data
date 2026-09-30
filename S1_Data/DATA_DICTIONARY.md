# Data dictionary

All identifiers are retained from the original corpus, including nonconsecutive pair IDs. Category `A_prime` corresponds to A′ in the article. Four `HN` pairs are unscored and never enter the 96-pair summaries.

| Field / table | Meaning |
|---|---|
| `benchmark[].pair_id`, `category`, `target_label` | Stable local record ID, design category, and construction target; not validated truth. |
| `construct_1/2.label` | Surface label used by the historical scalar mapping. It is not the entire construct description. |
| `definition_source` | Retained bibliographic/source string, used only by a documented string heuristic; not a verified quotation. |
| `definition_backfill.confidence/grounding` | Historical preparation flags. Missing record means unknown provenance, not source authenticity. |
| `curation.resolution.status` | Historical curation-status field. All retained records say `not_yet_curated`. |
| `panel[].panel_judgments` | Three model identities × three attempted repetitions for each of 96 pairs. |
| `judge`, `replicate` | API model identifier and repetition index 0–2. Repetitions are not independent judges. |
| `label_assigned`, `first_yes_question` | Parsed historical model label and first affirmative prompt question. |
| `error` | `null` for no recorded error, otherwise `RECORDED_ERROR`. Detailed error messages are excluded. |
| `systems[].raw_score` | Already normalized scalar similarity for S2/S5; no second cosine transformation is performed. |
| `systems[].predicted_label` | Parsed raw label for S6; absent for scalar systems until the mapping is applied. |
| `panel_pairs.csv` | Reconstructed panel assignments, modal votes, raw-response entropy, and availability. Two panel ties remain present. |
| `system_pair_agreement.csv` | One system prediction per pair; two panel ties excluded. `NONCONTROLS` is 73 resolved pairs; `ALL` is 94 and includes 21 calibration controls. |
| `q6_mcnemar.csv` | Exact matched-pair tests on 29 panel-L3 pairs, with monotone Holm adjustment over two comparisons. |
| `q7_entropy_permutation.csv` | Pair-level entropy contrasts; upper, lower, and absolute-difference two-sided tails are distinct. |
| `leave_one_judge_out*.csv` | Omission sensitivity; ties/insufficient votes are reported, not resolved arbitrarily. |
| `risk_recomputed.csv` | Historical priority/tier heuristics recomputed using the exact raw-file flags. `gold_tier` is a legacy name, not a validated reference set. |

Each CSV includes a header. Booleans are `True`/`False`; empty CSV cells represent missing fields. Lists and dictionaries in CSV cells use JSON encoding. JSON numeric values retain machine precision; manuscript percentages and coefficients are rounded for display.
