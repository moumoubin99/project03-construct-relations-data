# Data for prompt order and ontology relation judgments

This repository contains the deidentified numerical data and offline analysis code accompanying the PLOS ONE manuscript *Prompt order and definition inputs in large language model judgments of selected behavioural-science ontology relations*.

## Contents

- `S1_Data/`: historical construct-panel labels, scores, provenance metadata, prompt templates, reported tables, and numerical reproduction code. The numerical projection does not include original construct definitions, third-party measurement-item wording, free-text rationales, or the source rating workbooks.
- `S2_Data/`: the 66-item follow-up's initial and separately marked recovery outputs, prediction/reference codes, prompt definitions, and public-resource search audit. The historical source wording and original rating workbooks are not included.
- `S3_Data/`: 93 selected native BCIO subclass edges, the fixed request schedule, deidentified formal model-response journal, analysis code, and regenerated per-presentation and aggregate results.
- `MANIFEST_SHA256.txt`: hashes of every file in this release, except the manifest itself.

S1 and S2 numerical records allow the statistics and figures reported from those records to be recomputed. They do not independently verify the meaning of omitted historical wording. S3 permits offline regeneration of the public-ontology experiment's reported numerical results, including the 2,232 scheduled formal requests and their outcomes. No API key, GPU, or model endpoint is needed to run these analyses.

## Reproduce the numerical results

From the repository root:

```text
python3 S1_Data/reproduce_numeric.py --input S1_Data/numeric_inputs.json --output S1_Data/numeric_reproduced.json
python3 S2_Data/verify_numeric_data.py
python3 S3_Data/analyze_public.py --root S3_Data
```

The S1 command needs Python 3 and PyYAML. The S2 and S3 commands use the Python standard library. The S3 analyzer verifies the retained immutable scientific input files against the original experiment's frozen manifest, validates the deidentified journal, and then regenerates `S3_Data/analysis/metrics.json` and `S3_Data/analysis/per_cell.csv`.

## Deidentification and provenance

The public copy omits local runtime probes, code-review traces, transport/retry helpers, private source-file fingerprints, detailed upstream Git commit metadata, and superseded technical smoke journals. It removes local machine paths, service request identifiers, and access-token query parameters from retained records. The provider response envelopes were deidentified before regeneration of the S3 results. Scientific inputs, returned labels, outcome statuses, usage totals, and the 93 source assertions were preserved; the numerical results were checked against the original retained outputs. The journal file hash and any aggregate hash that binds to that file necessarily differ from the private original. `MANIFEST_SHA256.txt` records the public files' hashes.

The source for the selected native BCIO concepts and explicit subclass assertions is the Human Behaviour Change Project's Behaviour Change Intervention Ontology, Mechanism of Action module, version 11 September 2026, commit `1a839af717aa29dc85b8c875cb1c33d648b4d271`:

https://github.com/HumanBehaviourChangeProject/ontologies/tree/1a839af717aa29dc85b8c875cb1c33d648b4d271/MechanismOfAction

The selected BCIO content retains its CC BY 4.0 attribution and source terms. This release does not include the full ontology, Cognitive Atlas snapshots, or unrelated third-party measurement text. Historical resource links with token query parameters have had those parameters removed; identifiers and source descriptions remain where present.

The reported panel labels and single-rater reference codes represent historical judgments, not independently established correct construct relations. The public ontology task measures agreement with selected positive subclass assertions, not general relation-classification accuracy.

## Citation and reuse

Please cite the companion article and this version of the data if reused. Original numerical tables and metadata prepared by the study authors are shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). BCIO material retains its source attribution and CC BY 4.0 terms. Analysis scripts prepared for this release are available under the [MIT License](LICENSE-CODE.md). These statements do not grant rights to omitted third-party wording.
