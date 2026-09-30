# Exact analysis specification before new model outputs

This is a dated analysis plan, not external preregistration. REVIEW_ADOPTION.md supersedes the earlier draft's confirmatory language: p-values and intervals are conditional sensitivity results only.

## Statistical unit and estimands

The formal population for reporting is the supplied66-item set. For each of the three named providers and armsD/F/L, one initial response gives s_i=1 if a complete schema-valid label matches the independent human reference exactly, otherwise0. Never-dispatched cells are missing execution, not zero. No main-table result is released as complete unless all594 initial cells are present once. A request failure dispatched and completed as an error is zero and is reported separately. Six practice items, including R001 used for nine interface tests, never enter the66-item denominator.

Agreement A_a=sum_i s_i,a/66. Contrast Δ_F-D=sum_i(s_i,F−s_i,D)/66; likewise F-L. Substantive coverage counts predictionsC01–C11 over all attempts. Substantive correct resolution restricts the reference denominator toC01–C11; C12/C13-reference cases remain in the overall agreement table. Empty-class/subset rates are undefined, not100% or0%. Macro-F1 averages only classes present in the reference, with the exact class list reported.

## Dependency components

Use human_analysis/dependency_audit.json, built from the supplied family_groups.json: connect any two of the96 candidates sharing a listed broad construct family, take transitive closure BEFORE restricting to the formal66, then use that component membership. This prevents withheld or practice nodes from severing known connections. It yields33 components in the formal set. Components are conservative bookkeeping, not established statistical independence or construct equivalence. The full membership/size lists are frozen files. Retain the separate46-item sensitivity that excludes components connected to practice or historical pilot seeds.

For componentg, d_g=sum_{i in g}(s_i,F−s_i,B), n_g=number of formal items in g. Primary descriptive Δ=sum_g d_g/sum_g n_g. Component-equal sensitivity=(1/G)sum_g d_g/n_g. Leave-one-component-out effects=(sum_g d_g−d_h)/(N−n_h), report minimum and maximum.

## Conditional uncertainty, not confirmatory testing

Cluster bootstrap: drawG components with replacement10000times using NumPy default_rng(20260915), retain complete paired blocks, compute the ratio of sampled effect sums to sampled item counts. Report2.5th/97.5th percentiles as an assumption-conditional interval. This does not establish random-sampling validity for a purposive collection.

Sign-flip sensitivity: test statistic abs(sum_g d_g); independently multiply each nonzero component sum by±1. If K≤18 nonzero sums, enumerate2^K sign patterns. Otherwise sample100000 patterns with the same seeded generator following bootstrap draws and use(1+#statistics≥observed)/(100001). Zero-sum components can be omitted without changing the conditional distribution. Include all equal/extreme outcomes with numerical tolerance1e−12. Report K, exact/MonteCarlo mode, number of patterns/draws and grid resolution. Holm-adjust the six specified contrasts as one family, but adjusted values remain conditional/exploratory.

The sign procedure requires joint sign symmetry/exchangeability of component-level differences under the null. Mere component independence or a zero average effect does not imply it. We have not established those assumptions; no confirmatory “statistically proven benefit” follows from a small p-value. Decisions retain observed effects, their conditional precision, class-specific errors and sensitivity to removing components. Five percentage points is a pre-output descriptive practical threshold, not a verified detectable effect. A non-significant result does not establish equivalence.

## Error analysis

Report false equivalence predictions against non-C01 references (denominator61 in this set); false hierarchy predictionsC02–C05 against references outside that group (denominator57); exact agreement on the9 hierarchy-group references; and opposite-direction errorsC02↔C03,C04↔C05,C06↔C07. All nine positive hierarchy-group references are C04/C05 context restrictions. Do not report sensitivity for taxonomic type inclusionC02/C03, multiple-applicabilityC12 or insufficient-evidenceC13 because no formal reference has those labels.

All subset IDs and definitions are in human_analysis/subsets.json and fixed before calls. Do not pick a favorable subset after seeing results. Compare the old discussion summary only through explicitly conditional option correspondence; it is not the primary model ground truth.

## Execution checks

analyze_supplement.py and test_analysis.py implement these formulas; tests include independent toy counts, opposite directions, missing classes, invalid responses, C12/C13, component weighting, exact sign enumeration, zero effects and Holm. The runtime result adapter must independently align the actual initial response logs to unique provider/arm/item IDs; it may not silently retain a last duplicate or average practice cases into the formal set.
