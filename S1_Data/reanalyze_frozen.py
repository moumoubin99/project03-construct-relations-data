#!/usr/bin/env python3
"""Offline Project03 reanalysis. Python >=3.10 + existing PyYAML only.

No model clients, network, subprocesses, GPU, downloads, or legacy-code imports.
Inputs are read-only; all generated files must remain inside --output-dir.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import platform
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml

LABELS = tuple(f"L{i}" for i in range(1, 7))
CATEGORIES = ("A", "A_prime", "B", "C", "D")
RAW_FILES = (
    "BENCHMARK_v0.4.yaml", "label_panel_v1.6_chinese_full.yaml",
    "S6_kimi_direct_v1.6.json", "query_generic_embedding_v1.6.json",
    "query_wulff_mata_v1.6.json", "validation_risk_table.csv",
    "validation_risk_table.json",
)
SYSTEM_FILES = {
    "S2_wulff_mata": "query_wulff_mata_v1.6.json",
    "S5_generic_embed": "query_generic_embedding_v1.6.json",
    "S6_llm_direct": "S6_kimi_direct_v1.6.json",
}


def load(path):
    with path.open(encoding="utf-8-sig") as f:
        return yaml.safe_load(f) if path.suffix == ".yaml" else json.load(f)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("no_differences\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})


def valid_call(call, key="label_assigned"):
    return call.get("error") is None and call.get(key) in LABELS


def mode(values):
    counts = Counter(v for v in values if v is not None)
    candidates = sorted(k for k, n in counts.items() if n == max(counts.values())) if counts else []
    return (candidates[0] if len(candidates) == 1 else None, len(candidates) > 1, candidates)


def entropy(counts):
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n) if total else None


def aggregate(calls, judges):
    valid = [j for j in calls if valid_call(j)]
    modals, ties, candidates = {}, {}, {}
    for judge in judges:
        modals[judge], ties[judge], candidates[judge] = mode(
            [j["label_assigned"] for j in valid if j["judge"] == judge])
    winner, tied, across_candidates = mode(modals.values())
    panel_tie = sum(ties.values()) >= 2 or tied
    label = "TIE" if panel_tie else winner
    votes = dict(Counter(j["label_assigned"] for j in valid))
    fyq = dict(Counter(j["first_yes_question"] for j in valid if j.get("first_yes_question")))
    disagreement = (any(ties.values()) or panel_tie or
                    len(set(v for v in modals.values() if v is not None)) > 1 or
                    (winner is not None and any(j["label_assigned"] != winner for j in valid)))
    return {
        "majority_label": label, "majority_tie": panel_tie,
        "tie_candidates": [] if sum(ties.values()) >= 2 else (across_candidates if tied else []),
        "per_judge_modal": modals, "per_judge_modal_tie": ties,
        "per_judge_modal_candidates": candidates,
        "cross_judge_modal_distribution": dict(Counter(v for v in modals.values() if v is not None)),
        "vote_distribution": votes, "vote_entropy": entropy(votes),
        "first_yes_distribution": fyq, "first_yes_entropy": entropy(fyq),
        "disagreement_flag": bool(disagreement), "n_valid": len(valid), "n_attempts": len(calls),
        "n_valid_first_yes": sum(fyq.values()),
    }


def counts_row(values):
    values = list(values)
    return [sum(v == lab for v in values) for lab in LABELS]


def fleiss_kappa(rows):
    if not rows:
        return None
    n = sum(rows[0])
    if n < 2 or any(sum(r) != n for r in rows):
        raise ValueError("Fleiss kappa requires a constant number >=2 of raters per pair")
    total = len(rows) * n
    pbar = sum(sum(c * (c - 1) for c in r) / (n * (n - 1)) for r in rows) / len(rows)
    pe = sum((sum(r[k] for r in rows) / total) ** 2 for k in range(len(rows[0])))
    return (pbar - pe) / (1 - pe) if pe < 1 else None


def krippendorff_alpha(rows):
    rows = [r for r in rows if sum(r) >= 2]
    n = sum(sum(r) for r in rows)
    if n < 2:
        return None
    observed = sum((sum(r) ** 2 - sum(c * c for c in r)) / (sum(r) - 1) for r in rows) / n
    totals = [sum(r[k] for r in rows) for k in range(len(rows[0]))]
    expected = (n * n - sum(t * t for t in totals)) / (n * (n - 1))
    return 1 - observed / expected if expected > 0 else None


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    i = (len(values) - 1) * q
    low = math.floor(i)
    return values[low] + (values[math.ceil(i)] - values[low]) * (i - low)


def bootstrap_reliability(rows, n_judges, iterations, seed):
    rng = random.Random(seed)
    ks, als = [], []
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        k = fleiss_kappa([r for r in sample if sum(r) == n_judges])
        a = krippendorff_alpha(sample)
        if k is not None:
            ks.append(k)
        if a is not None:
            als.append(a)
    return {
        "method": "Unstratified pair bootstrap; resample all panel pairs with replacement, retain all judge modals for each selected pair; percentile 95% CI. Fleiss uses complete-judge pairs within each draw; alpha uses pairable pairs.",
        "iterations": iterations, "seed": seed, "rng": "Python random.Random (MT19937)",
        "fleiss_kappa_ci95": [quantile(ks, .025), quantile(ks, .975)],
        "krippendorff_alpha_ci95": [quantile(als, .025), quantile(als, .975)],
        "valid_fleiss_draws": len(ks), "valid_alpha_draws": len(als),
        "limitation": "Describes variation across this designed pair collection; does not resample judges or establish external population validity.",
    }


def mcnemar(a, b):
    if len(a) != len(b):
        raise ValueError("Matched pair lists must have equal length")
    cells = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    for x, y in zip(a, b):
        cells[f"n{int(x)}{int(y)}"] += 1
    discordant = cells["n01"] + cells["n10"]
    tail = sum(math.comb(discordant, k) for k in range(min(cells["n01"], cells["n10"]) + 1))
    p = min(1., 2 * tail / 2 ** discordant)
    return {"n_pairs": len(a), **cells, "discordant_pairs": discordant, "p_exact_two_sided": p}


def holm(p_values):
    adjusted, running_max = {}, 0.
    for rank, (key, p) in enumerate(sorted(p_values.items(), key=lambda x: (x[1], x[0]))):
        running_max = max(running_max, (len(p_values) - rank) * p)
        adjusted[key] = min(1., running_max)
    return adjusted


def permutation_difference(values, is_a, iterations, seed):
    n_a, n = sum(is_a), len(values)
    if not 0 < n_a < n:
        raise ValueError("Both permutation groups must be nonempty")
    total = sum(values)
    observed_sum = sum(v for v, a in zip(values, is_a) if a)
    difference = lambda s: s / n_a - (total - s) / (n - n_a)
    observed = difference(observed_sum)
    exact = math.comb(n, n_a) <= iterations
    rng = random.Random(seed)
    selections = (itertools.combinations(range(n), n_a) if exact else
                  (rng.sample(range(n), n_a) for _ in range(iterations)))
    null = [difference(sum(values[i] for i in chosen)) for chosen in selections]
    greater = sum(x >= observed - 1e-12 for x in null)
    smaller = sum(x <= observed + 1e-12 for x in null)
    two_sided = sum(abs(x) >= abs(observed) - 1e-12 for x in null)
    add = 0 if exact else 1
    denominator = len(null) + add
    return {
        "n_a_prime": n_a, "n_comparator": n - n_a,
        "mean_a_prime": observed_sum / n_a, "mean_comparator": (total - observed_sum) / (n - n_a),
        "observed_difference_bits": observed,
        "p_greater_primary": (greater + add) / denominator,
        "p_less_exploratory": (smaller + add) / denominator,
        "p_two_sided_exploratory": (two_sided + add) / denominator,
        "null_mean": statistics.mean(null), "null_sd": statistics.pstdev(null),
        "permutations": len(null), "exact_enumeration": exact, "seed": seed,
        "upper_tail_exceedances": greater, "lower_tail_exceedances": smaller,
        "two_sided_exceedances": two_sided, "monte_carlo_add_one": bool(add),
        "primary_mc_standard_error_approx": math.sqrt(((greater + add) / denominator) * (1 - (greater + add) / denominator) / denominator),
    }


def calibrate(records, benchmark):
    groups = {label: sorted(r["raw_score"] for r in records
                            if r.get("error") is None and r.get("raw_score") is not None
                            and benchmark[r["pair_id"]]["category"] == "D"
                            and benchmark[r["pair_id"]]["target_label"] == label)
              for label in ("L1", "L4")}
    defaults = {"L1_min": .90, "L2_min": .75, "L3_min": .55}
    meta = {"n_D_L1": len(groups["L1"]), "n_D_L4": len(groups["L4"]),
            "method": "Archived common.py rule: upper order-statistic median (including even n), margin 0.03, fitted on designed D control labels before panel scoring. No tuning in this reanalysis."}
    if min(map(len, groups.values())) < 3:
        return defaults, {**meta, "fallback": "insufficient controls"}
    same, diff = (groups[k][len(groups[k]) // 2] for k in ("L1", "L4"))
    high, low = max(0., same - .03), min(1., diff + .03)
    if low >= high:
        return defaults, {**meta, "fallback": "degenerate control separation"}
    return {"L1_min": high, "L2_min": (high + low) / 2, "L3_min": low}, {
        **meta, "same_upper_median": same, "different_upper_median": diff, "fallback": None}


def normalize(score, pair, thresholds):
    same = pair["construct_1"]["label"].strip().lower() == pair["construct_2"]["label"].strip().lower()
    if score >= thresholds["L1_min"]:
        return "L1"
    if score >= thresholds["L2_min"]:
        return "L2" if same else "L5"
    if score >= thresholds["L3_min"]:
        return "L2"
    return "L4"


# Preserved string heuristics from build_validation_risk.py, used for comparison
# only. These are not bibliographic verification or human validation.
BOOK_MARKERS = ["handbook", "press)", " press", "wiley", "sage", "routledge", "guilford",
                "lexington book", "(ed", "(eds", "chapter", "encyclopedia", "monograph",
                "van nostrand", "addison-wesley", "basic books", "free press", "doubleday",
                "the good soldier", "transparent self", "managing brand equity",
                "flow: the psychology", "ecological approach", "psychology of everyday",
                "advances in experimental"]
JOURNAL_MARKERS = ["journal", "review", "quarterly", "psychological", "human factors",
                   "research", "science", "misq", "jasist", "annual review", "proceedings",
                   "educ. psychol. meas", "personality", "marketing", "management review",
                   "behavior", "bulletin", "psychometrika", "applied psychology"]


def source_heuristic(src):
    if not src or not str(src).strip():
        return "weak"
    s = str(src).lower()
    if any(x in s for x in ("label-only", "no source", "not available")):
        return "weak"
    doi = "doi:" in s or "doi.org" in s or bool(re.search(r"10\.\d{4}", s))
    year = bool(re.search(r"\b(19|20)\d{2}\b", s))
    book = any(x in s for x in BOOK_MARKERS)
    journal = any(x in s for x in JOURNAL_MARKERS)
    if not book and (doi or (journal and year)):
        return "strong"
    return "medium" if book or year else "weak"


def risk_row(bp, panel):
    sides = [bp[k] for k in ("construct_1", "construct_2")]
    backfills = [s["definition_backfill"] for s in sides if s.get("definition_backfill")]
    n_backfilled = len(backfills)
    status = ("both-sides-original", "one-side-backfilled", "both-sides-backfilled")[n_backfilled]
    confs = [x["confidence"] for x in backfills if x.get("confidence")]
    conf_rank = {"high": 3, "medium": 2, "low": 1, "irreducibly-label-only": 0}
    min_conf = min(confs, key=lambda c: conf_rank.get(c, 99)) if confs else "none"
    strength = min((source_heuristic(s.get("definition_source")) for s in sides),
                   key=lambda s: {"strong": 2, "medium": 1, "weak": 0}[s])
    disagreement = panel["disagreement_flag"] or panel["majority_tie"] or panel["vote_entropy"] >= .6
    unanimous = panel["vote_entropy"] == 0 and not panel["disagreement_flag"] and not panel["majority_tie"]
    operational = (not n_backfilled or min_conf == "high") and strength in ("strong", "medium") and unanimous
    strict = n_backfilled == 0 and unanimous and strength == "strong"
    priority = ("HIGH" if (n_backfilled and disagreement) or strength == "weak" else
                "LOW" if operational and strength == "strong" and min_conf in ("none", "high") else "MED")
    return {
        "pair_id": bp["pair_id"], "category": bp["category"], "panel_majority_label": panel["majority_label"],
        "vote_entropy": panel["vote_entropy"], "first_yes_entropy": panel["first_yes_entropy"],
        "disagreement_flag": panel["disagreement_flag"], "majority_tie": panel["majority_tie"],
        "backfill_status": status, "n_sides_backfilled": n_backfilled,
        "min_backfill_confidence": min_conf, "source_strength": strength,
        "high_disagreement": bool(disagreement), "unanimous_panel": unanimous,
        "human_validation_priority": priority, "gold_tier": bool(operational), "gold_tier_strict": bool(strict),
        "no_backfill_record_sides": 2 - n_backfilled,
        "construct_1_origin": (sides[0].get("definition_backfill") or {}).get("grounding", "no-backfill-record"),
        "construct_2_origin": (sides[1].get("definition_backfill") or {}).get("grounding", "no-backfill-record"),
        "construct_1_definition_source": sides[0].get("definition_source"),
        "construct_2_definition_source": sides[1].get("definition_source"),
    }


def compare_field(pid, field, old, new, source, diffs):
    equal = abs(old - new) <= .000051 if (isinstance(old, (int, float)) and not isinstance(old, bool)
            and isinstance(new, (int, float)) and not isinstance(new, bool)) else old == new
    if not equal:
        diffs.append({"source": source, "pair_id": pid, "field": field, "archived": old, "recomputed": new})


def tally(rows, key):
    return dict(sorted(Counter(r[key] for r in rows).items(), key=lambda x: str(x[0])))


def origin_summary(pairs):
    sides = [p[s] for p in pairs for s in ("construct_1", "construct_2")]
    n_bf = sum(bool(s.get("definition_backfill")) for s in sides)
    return {"n_pairs": len(pairs), "n_sides": len(sides), "backfilled_sides": n_bf,
            "no_backfill_record_sides": len(sides) - n_bf,
            "backfilled_sides_percent": 100 * n_bf / len(sides) if sides else None,
            "side_grounding_counts": dict(Counter((s.get("definition_backfill") or {}).get("grounding", "no-backfill-record") for s in sides)),
            "pair_n_backfilled_counts": dict(sorted(Counter(sum(bool(p[s].get("definition_backfill")) for s in ("construct_1", "construct_2")) for p in pairs).items())),
            "side_backfill_confidence_counts": dict(Counter(s["definition_backfill"].get("confidence", "not-recorded") for s in sides if s.get("definition_backfill"))),
            "curation_status_counts": dict(Counter(p.get("curation", {}).get("resolution", {}).get("status", "not-recorded") for p in pairs))}


def run(data, out, bootstraps=5000, permutations=49999, seed=20260914):
    data, out = data.resolve(), out.resolve()
    if out == data or data in out.parents or out in data.parents:
        raise ValueError("Output and raw data directories must be separate, nonnested directories")
    out.mkdir(parents=True, exist_ok=True)
    inputs = {name: {"sha256": sha256(data / name), "bytes": (data / name).stat().st_size}
              for name in RAW_FILES}
    bench_doc, panel_doc = (load(data / name) for name in RAW_FILES[:2])
    benchmark = {p["pair_id"]: p for p in bench_doc["pairs"]}
    assert len(benchmark) == len(bench_doc["pairs"]) == 100
    scored = {k: p for k, p in benchmark.items() if not p.get("unscored") and p["category"] != "HN"}
    panel_raw = {p["pair_id"]: p for p in panel_doc["pairs"]}
    assert len(panel_raw) == len(panel_doc["pairs"]) == 96 and set(panel_raw) == set(scored)
    judges = [j["alias_label"] for j in panel_doc["metadata"]["judges"]]
    assert len(judges) == 3 and panel_doc["metadata"]["replicates"] == 3
    panels, calls, differences, judge_rows = {}, [], [], []
    for pid, raw in panel_raw.items():
        assert raw["category"] == scored[pid]["category"]
        js = raw["panel_judgments"]
        assert {(j["judge"], j["replicate"]) for j in js} == set(itertools.product(judges, range(3))) and len(js) == 9
        current = aggregate(js, judges)
        current["judge_modal_entropy_bits"] = entropy(current["cross_judge_modal_distribution"])
        current["n_judge_modals_for_entropy"] = sum(current["cross_judge_modal_distribution"].values())
        panels[pid] = {"pair_id": pid, "category": scored[pid]["category"], **current}
        for j in js:
            calls.append({"pair_id": pid, "category": scored[pid]["category"], "judge": j["judge"],
                          "replicate": j["replicate"], "label_assigned": j.get("label_assigned"),
                          "first_yes_question": j.get("first_yes_question"), "error": j.get("error"),
                          "valid": valid_call(j)})
        for judge in judges:
            sub = [j for j in js if j["judge"] == judge]
            valid = [j for j in sub if valid_call(j)]
            judge_rows.append({"pair_id": pid, "category": scored[pid]["category"], "judge": judge,
                               "n_attempts": len(sub), "n_valid": len(valid),
                               "n_L4": sum(j["label_assigned"] == "L4" for j in valid),
                               "modal_label": current["per_judge_modal"][judge],
                               "modal_tie": current["per_judge_modal_tie"][judge],
                               "modal_candidates": current["per_judge_modal_candidates"][judge],
                               "replicate_unanimity_evaluable": len(valid) >= 2,
                               "replicate_unanimous": len({j["label_assigned"] for j in valid}) == 1 if len(valid) >= 2 else None})
        for field in ("majority_label", "majority_tie", "per_judge_modal", "per_judge_modal_tie",
                      "vote_distribution", "vote_entropy", "first_yes_distribution", "disagreement_flag"):
            compare_field(pid, field, raw["aggregation"].get(field, False if field == "majority_tie" else None),
                          current[field], "panel_yaml_aggregation", differences)
    assert len(calls) == 864 and sum(c["valid"] for c in calls) == 847
    ids = sorted(panels)
    rating_rows = [counts_row(panels[pid]["per_judge_modal"].values()) for pid in ids]
    reliability = {
        "unit": "pair; nominal L1-L6; one untied modal label per judge",
        "fleiss_kappa": fleiss_kappa([r for r in rating_rows if sum(r) == len(judges)]),
        "krippendorff_alpha_nominal": krippendorff_alpha(rating_rows),
        "fleiss_complete_pairs": sum(sum(r) == len(judges) for r in rating_rows),
        "alpha_pairable_pairs": sum(sum(r) >= 2 for r in rating_rows),
        "available_modal_judges_per_pair_counts": dict(Counter(sum(r) for r in rating_rows)),
        "bootstrap": bootstrap_reliability(rating_rows, len(judges), bootstraps, seed),
    }
    legacy_tiebreak_rows = [counts_row(
        (panels[pid]["per_judge_modal_candidates"][j] or [None])[0] for j in judges) for pid in ids]
    reliability["legacy_lexicographic_tiebreak_sensitivity"] = {
        "role": "Comparison reconstructed from raw calls; not the primary tie policy",
        "fleiss_kappa": fleiss_kappa([r for r in legacy_tiebreak_rows if sum(r) == len(judges)]),
        "krippendorff_alpha_nominal": krippendorff_alpha(legacy_tiebreak_rows),
        "fleiss_complete_pairs": sum(sum(r) == len(judges) for r in legacy_tiebreak_rows),
        "affected_cells": [{"pair_id": row["pair_id"], "judge": row["judge"],
                            "candidates": row["modal_candidates"]} for row in judge_rows if row["modal_tie"]],
        "cause": "analyze_results.py assigns winners[0] to tied within-judge modes; run_label_panel.py excludes those modes. Primary analysis follows the latter consistently.",
    }
    raw_rates = []
    for judge in ["ALL", *judges]:
        for category in ["ALL", *CATEGORIES]:
            sub = [c for c in calls if (judge == "ALL" or c["judge"] == judge)
                   and (category == "ALL" or c["category"] == category)]
            valid = [c for c in sub if c["valid"]]
            n4 = sum(c["label_assigned"] == "L4" for c in valid)
            raw_rates.append({"judge": judge, "category": category, "attempts": len(sub), "valid": len(valid),
                              "invalid": len(sub) - len(valid), "n_L4": n4,
                              "L4_rate_valid_calls": n4 / len(valid) if valid else None,
                              "L4_rate_all_attempts_descriptive": n4 / len(sub) if sub else None,
                              "error_count": sum(c["error"] is not None for c in sub),
                              "invalid_label_without_error": sum(c["error"] is None and not c["valid"] for c in sub)})
    category_rows = []
    for cat in ["ALL", *CATEGORIES]:
        ps = [p for p in panels.values() if cat == "ALL" or p["category"] == cat]
        category_rows.append({"category": cat, "n_pairs": len(ps),
                              "n_disagreement": sum(p["disagreement_flag"] for p in ps),
                              "n_panel_ties": sum(p["majority_tie"] for p in ps),
                              "n_unanimous_valid_calls": sum(p["vote_entropy"] == 0 for p in ps),
                              "mean_vote_entropy_bits": statistics.mean(p["vote_entropy"] for p in ps),
                              "mean_first_yes_entropy_bits": statistics.mean(p["first_yes_entropy"] for p in ps),
                              "mean_judge_modal_entropy_bits": statistics.mean(p["judge_modal_entropy_bits"] for p in ps)})
    system_pairs, system_calls, calibrations, system_maps, system_identity = [], [], {}, {}, {}
    for system, filename in SYSTEM_FILES.items():
        records = load(data / filename)
        assert all(r["system_id"] == system and r["pair_id"] in scored
                   and r["category"] == scored[r["pair_id"]]["category"] for r in records)
        assert len({(r["pair_id"], r["replicate_idx"]) for r in records}) == len(records)
        assert set(r["pair_id"] for r in records) == set(scored)
        assert len(records) == (288 if system == "S6_llm_direct" else 96)
        system_identity[system] = {
            "file": filename, "records": len(records), "pairs": len(set(r["pair_id"] for r in records)),
            "input_fields": (["label", "definition", "items_sample"] if system == "S2_wulff_mata" else
                             ["label", "definition"] if system == "S5_generic_embed" else
                             ["label", "discipline", "definition", "items_sample"]),
            "raw_payload_counts": {field: dict(Counter(r.get("raw_payload", {}).get(field, "not-recorded") for r in records))
                                   for field in (("embed_backend", "embed_model", "text_variant") if system != "S6_llm_direct" else ("backend", "model", "prompt_mode"))},
            "input_rule_evidence": ("query_wulff_mata.evaluate_pair -> common.pair_text_for_embedding: newline-joined label, definition, optional pipe-joined items; no discipline field" if system == "S2_wulff_mata" else
                                    "query_generic_embedding._text: newline-joined label and definition, explicitly excludes items and discipline" if system == "S5_generic_embed" else
                                    "query_llm_direct.S6_DIRECT_USER_TEMPLATE and _format_user_prompt: label, discipline, definition, and items for both constructs"),
        }
        if system == "S2_wulff_mata":
            text_length_mismatches = []
            for record in records:
                for index in (1, 2):
                    construct = scored[record["pair_id"]][f"construct_{index}"]
                    parts = [construct["label"]]
                    if construct.get("definition"):
                        parts.append(construct["definition"])
                    if construct.get("items_sample"):
                        parts.append(" | ".join(construct["items_sample"]))
                    rebuilt_length = len("\n".join(p for p in parts if p))
                    archived_length = record.get("raw_payload", {}).get(f"text_{index}_len")
                    if archived_length != rebuilt_length:
                        text_length_mismatches.append({"pair_id": record["pair_id"], "side": index,
                                                       "archived_length": archived_length, "rebuilt_length": rebuilt_length})
            system_identity[system]["reconstructed_input_length_mismatches"] = text_length_mismatches
        thresholds = None
        if system != "S6_llm_direct":
            thresholds, meta = calibrate(records, scored)
            calibrations[system] = {"thresholds": thresholds, **meta}
        grouped = defaultdict(list)
        for r in records:
            pid = r["pair_id"]
            label = r.get("predicted_label") if r.get("error") is None else None
            derived = False
            if label is None and thresholds is not None and r.get("error") is None and r.get("raw_score") is not None:
                label = normalize(r["raw_score"], scored[pid], thresholds)
                derived = True
            valid = label in LABELS
            row = {"system": system, "pair_id": pid, "category": scored[pid]["category"],
                   "replicate_idx": r["replicate_idx"], "raw_score": r.get("raw_score"),
                   "raw_predicted_label": r.get("predicted_label"), "derived_predicted_label": label,
                   "derived_from_frozen_score": derived, "valid": valid, "error": r.get("error")}
            system_calls.append(row)
            grouped[pid].append(row)
        mapping = {}
        for pid in ids:
            sub = grouped[pid]
            lab, tied, candidates = mode([r["derived_predicted_label"] for r in sub if r["valid"]])
            target = panels[pid]["majority_label"]
            mapping[pid] = lab
            system_pairs.append({"system": system, "pair_id": pid, "category": scored[pid]["category"],
                                 "n_attempts": len(sub), "n_valid": sum(r["valid"] for r in sub),
                                 "prediction": lab, "system_modal_tie": tied, "modal_candidates": candidates,
                                 "panel_label": target, "panel_tie": panels[pid]["majority_tie"],
                                 "agreement": lab == target if lab in LABELS and target in LABELS else None})
        system_maps[system] = mapping
    agreements = []
    for system in SYSTEM_FILES:
        for cat in ["ALL", *CATEGORIES, "NONCONTROLS"]:
            ps = [p for p in system_pairs if p["system"] == system and (cat == "ALL" or p["category"] == cat
                  or (cat == "NONCONTROLS" and p["category"] != "D"))]
            eligible = [p for p in ps if p["panel_label"] in LABELS]
            evaluable = [p for p in eligible if p["prediction"] in LABELS]
            matches = sum(p["agreement"] for p in evaluable)
            agreements.append({"system": system, "category": cat, "n_pairs": len(ps),
                               "reporting_role": "primary" if cat == "NONCONTROLS" else "secondary_includes_calibration_controls" if cat == "ALL" else "category_descriptive",
                               "n_panel_ties_excluded": sum(p["panel_tie"] for p in ps),
                               "n_panel_missing_excluded": sum(p["panel_label"] is None for p in ps),
                               "n_system_unresolved": len(eligible) - len(evaluable),
                               "n_evaluable_pairs": len(evaluable), "n_agree": matches,
                               "pair_majority_agreement": matches / len(evaluable) if evaluable else None,
                               "agreement_if_system_unresolved_is_failure": matches / len(eligible) if eligible else None})
    system_call_rates = []
    for system in SYSTEM_FILES:
        for cat in ["ALL", *CATEGORIES, "NONCONTROLS"]:
            sub = [r for r in system_calls if r["system"] == system and
                   (cat == "ALL" or r["category"] == cat or (cat == "NONCONTROLS" and r["category"] != "D"))]
            native = [r for r in sub if r["error"] is None and r["raw_predicted_label"] in LABELS]
            usable = [r for r in sub if r["valid"]]
            system_call_rates.append({"system": system, "category": cat, "n_records": len(sub),
                                      "n_native_label_calls": len(native),
                                      "n_native_L4": sum(r["raw_predicted_label"] == "L4" for r in native),
                                      "native_call_L4_rate": sum(r["raw_predicted_label"] == "L4" for r in native) / len(native) if native else None,
                                      "n_usable_native_or_derived_labels": len(usable),
                                      "n_native_or_derived_L4": sum(r["derived_predicted_label"] == "L4" for r in usable),
                                      "native_or_derived_L4_rate": sum(r["derived_predicted_label"] == "L4" for r in usable) / len(usable) if usable else None,
                                      "label_origin": "native LLM labels" if system == "S6_llm_direct" else "offline calibrated scalar scores; raw file has no emitted labels"})
    q6, q6_pairs = {}, []
    for other in ("S2_wulff_mata", "S5_generic_embed"):
        shared = [pid for pid in ids if panels[pid]["majority_label"] == "L3"
                  and system_maps["S6_llm_direct"][pid] in LABELS and system_maps[other][pid] in LABELS]
        a = [system_maps["S6_llm_direct"][pid] == "L3" for pid in shared]
        b = [system_maps[other][pid] == "L3" for pid in shared]
        key = "S6_vs_" + other.split("_")[0]
        q6[key] = {**mcnemar(a, b), "matched_pair_ids": shared,
                   "s6_agreements": sum(a), "other_agreements": sum(b),
                   "agreement_difference": (sum(a) - sum(b)) / len(shared) if shared else None}
        q6_pairs.extend({"comparison": key, "pair_id": pid, "panel_label": "L3",
                         "s6_prediction": system_maps["S6_llm_direct"][pid], "other_prediction": system_maps[other][pid],
                         "s6_agrees": x, "other_agrees": y} for pid, x, y in zip(shared, a, b))
    adjusted = holm({key: v["p_exact_two_sided"] for key, v in q6.items()})
    for key in q6:
        q6[key]["p_holm"] = adjusted[key]
    q7 = {}
    for metric in ("first_yes_entropy", "vote_entropy", "judge_modal_entropy_bits"):
        for noncontrols in (False, True):
            subset = [panels[pid] for pid in ids if not noncontrols or panels[pid]["category"] != "D"]
            means = {cat: statistics.mean(p[metric] for p in subset if p["category"] == cat)
                     for cat in CATEGORIES if any(p["category"] == cat for p in subset)}
            key = metric + ("_noncontrols" if noncontrols else "_all")
            q7[key] = {**permutation_difference([p[metric] for p in subset],
                        [p["category"] == "A_prime" for p in subset], permutations, seed),
                       "per_category_mean_entropy_bits": means,
                       "a_prime_rank_descending": 1 + sum(v > means["A_prime"] for v in means.values()),
                       "pair_ids": [p["pair_id"] for p in subset]}
    lojo = []
    for pid in ids:
        p = panels[pid]
        for omitted in judges:
            remaining = [p["per_judge_modal"][j] for j in judges if j != omitted]
            supported = [x for x in remaining if x is not None]
            label, tied, candidates = mode(supported)
            verdict = "INSUFFICIENT" if len(supported) < 2 else "TIE" if tied else label
            status = ("full_panel_unresolved" if p["majority_label"] not in LABELS else
                      "unchanged" if verdict == p["majority_label"] else
                      "became_tie" if verdict == "TIE" else
                      "insufficient" if verdict == "INSUFFICIENT" else "changed_label")
            lojo.append({"pair_id": pid, "category": p["category"], "omitted_judge": omitted,
                         "full_panel_label": p["majority_label"], "remaining_judge_modals": remaining,
                         "lojo_label": verdict, "n_remaining_usable_judges": len(supported), "status": status})
    lojo_summary = {"counts_over_pair_omission_runs": tally(lojo, "status"),
                    "all_three_omissions_unchanged_pairs": sum(all(r["status"] == "unchanged" for r in lojo if r["pair_id"] == pid) for pid in ids),
                    "full_panel_resolved_pairs": sum(p["majority_label"] in LABELS for p in panels.values()),
                    "rule": "Omit one judge, retain each other judge's raw-replicate modal; two usable remaining judges must agree. Disagreement is TIE; fewer than two usable judges is INSUFFICIENT. No arbitrary tie break.",
                    "per_category": {cat: tally([r for r in lojo if r["category"] == cat], "status") for cat in CATEGORIES}}
    lojo_pair_summary = [{"pair_id": pid, "category": panels[pid]["category"],
                          "full_panel_label": panels[pid]["majority_label"],
                          "all_omissions_unchanged": all(r["status"] == "unchanged" for r in lojo if r["pair_id"] == pid),
                          **{status: sum(r["status"] == status for r in lojo if r["pair_id"] == pid)
                             for status in ("unchanged", "became_tie", "changed_label", "insufficient", "full_panel_unresolved")}}
                         for pid in ids]
    risks = [risk_row(scored[pid], panels[pid]) for pid in ids]
    old_risk = load(data / "validation_risk_table.json")
    with (data / "validation_risk_table.csv").open(encoding="utf-8-sig", newline="") as f:
        old_csv_rows = list(csv.DictReader(f))
    old_risk_map = {r["pair_id"]: r for r in old_risk["pairs"]}
    old_csv = {r["pair_id"]: r for r in old_csv_rows}
    assert len(old_risk_map) == len(old_risk["pairs"]) == 96 and set(old_risk_map) == set(scored)
    assert len(old_csv) == len(old_csv_rows) == 96 and set(old_csv) == set(scored)
    for row in risks:
        pid = row["pair_id"]
        for field, new in row.items():
            if field in old_risk_map[pid]:
                compare_field(pid, field, old_risk_map[pid][field], new, "validation_risk_json", differences)
            if field in old_csv[pid]:
                old = old_csv[pid][field]
                if isinstance(new, bool):
                    old = old.lower() == "true"
                elif isinstance(new, int):
                    old = int(old)
                elif isinstance(new, float):
                    old = float(old)
                compare_field(pid, field, old, new, "validation_risk_csv", differences)
    risk_categories = []
    for cat in ["ALL", *CATEGORIES]:
        rs = [r for r in risks if cat == "ALL" or r["category"] == cat]
        risk_categories.append({"category": cat, "n_pairs": len(rs),
                                "n_pairs_no_backfill_record_both_sides": sum(r["n_sides_backfilled"] == 0 for r in rs),
                                "n_backfilled_sides": sum(r["n_sides_backfilled"] for r in rs),
                                "pct_pairs_no_backfill_record_both_sides": 100 * sum(r["n_sides_backfilled"] == 0 for r in rs) / len(rs),
                                "mean_vote_entropy_bits": statistics.mean(r["vote_entropy"] for r in rs),
                                **{f"n_{p}": sum(r["human_validation_priority"] == p for r in rs) for p in ("HIGH", "MED", "LOW")},
                                "n_operational_tier": sum(r["gold_tier"] for r in rs),
                                "n_strict_tier": sum(r["gold_tier_strict"] for r in rs)})
    risk_summary = {"priority_counts": tally(risks, "human_validation_priority"),
                    "backfill_status_counts": tally(risks, "backfill_status"),
                    "source_strength_counts": tally(risks, "source_strength"),
                    "min_backfill_confidence_counts": tally(risks, "min_backfill_confidence"),
                    "n_gold_tier": sum(r["gold_tier"] for r in risks),
                    "n_gold_tier_strict": sum(r["gold_tier_strict"] for r in risks),
                    "n_provisional": sum(not r["gold_tier"] for r in risks)}
    for field, value in risk_summary.items():
        compare_field("SUMMARY", field, old_risk["summary"].get(field), value, "validation_risk_json_summary", differences)
    origins = {"all_100_pairs": origin_summary(list(benchmark.values())),
               "scored_96_pairs": origin_summary(list(scored.values())),
               "excluded_HN_4_pairs": origin_summary([p for pid, p in benchmark.items() if pid not in scored]),
               "scored_per_category": {cat: origin_summary([p for p in scored.values() if p["category"] == cat]) for cat in CATEGORIES},
               "interpretation": "No definition_backfill record is an absence-of-metadata category, not proof of original cited text or human validation. Grounding/confidence are archived metadata, not independently verified here."}
    legacy = {}
    legacy_file = data / "FINAL_analysis_v1.6.json"
    if legacy_file.exists():
        inputs[legacy_file.name] = {"sha256": sha256(legacy_file), "bytes": legacy_file.stat().st_size,
                                    "role": "comparison only, never computational input"}
        old = load(legacy_file)
        legacy = {"old_Q6": old.get("Q6"), "old_Q7": old.get("Q7"),
                  "old_per_system_graded": old.get("per_system_graded"),
                  "old_reliability": {k: old.get("panel_disagreement", {}).get(k)
                                      for k in ("fleiss_alpha", "krippendorff_alpha")}}
    evidence = {}
    for name in ("analyze_results.py", "run_label_panel.py", "common.py", "build_validation_risk.py", "query_llm_direct.py", "query_generic_embedding.py", "query_wulff_mata.py"):
        path = data.parent / "code" / "eval_scripts" / name
        if path.exists():
            evidence[name] = {"sha256": sha256(path), "role": "read-only rule evidence; not imported or executed"}
    prompt_source = data.parent / "code" / "eval_scripts" / "query_llm_direct.py"
    prompt_evidence = {"source_available": prompt_source.exists()}
    if prompt_source.exists():
        tree = ast.parse(prompt_source.read_text(encoding="utf-8-sig"))
        names = {"USER_TEMPLATE", "S6_DIRECT_USER_TEMPLATE"}
        templates = {target.id: (ast.literal_eval(node.value), node.lineno)
                     for node in tree.body if isinstance(node, ast.Assign)
                     for target in node.targets if isinstance(target, ast.Name) and target.id in names}
        assert set(templates) == names
        for name, (template, line) in templates.items():
            assert all("{" + field + "_" + side + "}" in template
                       for field in ("label", "disc", "def", "items") for side in ("1", "2"))
            prompt_evidence[name] = {"line": line, "both_construct_fields": ["label", "discipline", "definition", "items"]}
        prompt_evidence["formatting"] = "_format_user_prompt fills items from items_sample, separated by ' | ', or '(none provided)' when empty. Both source templates include all four fields; no new prompts were run."
    s6_raw = load(data / SYSTEM_FILES["S6_llm_direct"])
    prompt_evidence["frozen_S6_payload_counts"] = {
        field: dict(Counter(r.get("raw_payload", {}).get(field, "not-recorded") for r in s6_raw))
        for field in ("backend", "model", "prompt_mode")}
    assert all(sha256(data / name) == meta["sha256"] for name, meta in inputs.items())
    result = {
        "schema_version": "1.0", "scope": "Bounded offline reanalysis of frozen Project03 results; no new model or human judgments",
        "runtime": {"python": platform.python_version(), "PyYAML": yaml.__version__,
                    "dependencies": ["Python standard library", "PyYAML"], "seed": seed},
        "inputs": inputs, "rule_evidence": evidence,
        "panel": {"n_pairs": len(panels), "n_attempts": len(calls), "n_valid": sum(c["valid"] for c in calls),
                  "n_invalid": sum(not c["valid"] for c in calls), "judges": judges,
                  "valid_call_rule": "error is None and label_assigned is one of L1..L6; no retries or raw-response reparsing",
                  "entropy_definitions": {"vote_entropy": "Shannon entropy -sum p*log2(p) of labels across all valid raw calls in each pair; bits, not normalized. Calls carry equal weight; missing calls reduce a judge's contribution.",
                                          "first_yes_entropy": "Same raw-valid-call calculation for first_yes_question; empty first-YES fields excluded; bits, not normalized. Separate from modal reliability.",
                                          "judge_modal_entropy_bits": "Sensitivity: Shannon entropy of one untied modal label per available judge (equal judge weight). Missing and tied judge modes excluded; denominator 3 for 93 pairs and 2 for 3 pairs. Bits, not divided by log2(3)."},
                  "aggregation_rule": "Unique per-judge modal across valid replicates; tied modes excluded. Across retained judge modals use unique mode; >=2 tied judges or tied across-judge modes produce TIE. Matches archived rule, checked against stored aggregation only after reconstruction.",
                  "majority_label_counts": tally(list(panels.values()), "majority_label"),
                  "tie_pair_ids": [pid for pid in ids if panels[pid]["majority_tie"]],
                  "per_judge_modal_tie_cells": sum(r["modal_tie"] for r in judge_rows),
                  "per_judge_modal_missing_cells": sum(r["modal_label"] is None for r in judge_rows),
                  "first_yes_missing_on_valid_calls": sum(c["valid"] and not c["first_yes_question"] for c in calls),
                  "category_summary": category_rows, "reliability": reliability, "raw_call_L4": raw_rates,
                  "replicate_consistency": {j: {"evaluable_cells": sum(r["replicate_unanimity_evaluable"] for r in judge_rows if r["judge"] == j),
                                                       "unanimous_cells": sum(r["replicate_unanimous"] is True for r in judge_rows if r["judge"] == j)} for j in judges}},
        "systems": {"aggregation": "One unique modal prediction per pair (S6: three calls; S2/S5: one calibrated frozen score). Panel ties excluded. Unresolved system modes reported explicitly and excluded from evaluable denominator; failure-counting sensitivity supplied.",
                    "identity_file_input_mapping": system_identity,
                    "primary_noncontrol_agreement": [r for r in agreements if r["category"] == "NONCONTROLS"],
                    "secondary_all_pair_agreement": [r for r in agreements if r["category"] == "ALL"],
                    "prompt_input_evidence": prompt_evidence,
                    "call_level_L4_rates": system_call_rates,
                    "calibration": calibrations, "pair_majority_agreement": agreements,
                    "limitations": "S2 and S5 use BGE-M3 frozen approximations, not independent model families. Their archived scalar mapping cannot emit L3 or L6. D controls calibrate thresholds and are reused in the ALL score; NONCONTROLS is reported separately. Panel agreement is not validated accuracy. No raw outputs for S1/S3/S4 are in this bounded input set."},
        "Q6": {"method": "Exact two-sided matched-pair McNemar, S6 vs S2 and S6 vs S5, restricted to reconstructed panel L3; one modal prediction per pair. Holm over exactly two tests: cumulative maximum of sorted (m-rank)*p, capped at 1.",
               "comparisons": q6, "limitation": "The two scalar baselines cannot emit L3 under the archived mapping. A significant difference therefore includes a structural output-space advantage; it does not isolate hierarchy reasoning."},
        "Q7": {"method": "Difference in mean per-pair Shannon first-YES entropy (A_prime minus comparator), in bits. Randomly assign the A_prime membership across whole pairs at its fixed size, equivalent for this statistic to shuffling all category labels preserving category sizes. Monte Carlo p=(exceedances+1)/(B+1); exact enumeration used only if all allocations fit the budget.",
               "primary_direction": "A_prime has GREATER entropy than all non-A_prime pairs (one-sided upper tail).",
               "sensitivity": "Remove D calibration/control pairs: A_prime vs A+B+C. Raw label-vote entropy repeats the calculation as a measurement sensitivity. judge_modal_entropy_bits uses one available untied modal label per judge to remove replicate weighting. Lower-tail and absolute two-sided p-values are exploratory, not the pre-stated directional Q7.",
               "limitation": "Permutation reference assumes exchangeable pair category assignments; these categories are designed, not randomized. Resample entire pairs, never individual calls. No block by construct identity is available in this analysis, so reused constructs may violate pair independence. Failure of the greater-entropy prediction is not an equivalence test or proof of stability.",
               "results": q7},
        "leave_one_judge_out": lojo_summary, "S1_origin_counts": origins,
        "risk": {"current_summary": risk_summary, "per_category": risk_categories,
                 "interpretation": "Historical HIGH/MED/LOW and gold_tier flags are reproduced as sampling heuristics, not validated quality or correctness. Weakest definition_source string governs source_strength; a weak side does not imply both sides have no source. No source lookup or human validation was conducted.",
                 "archived_risk_pair_field_differences": sum(d["source"] in ("validation_risk_json", "validation_risk_csv") for d in differences)},
        "source_comparisons": {"differences": differences, "legacy_comparison_only": legacy},
        "verification": {"raw_input_hashes_unchanged": True, "unique_pair_and_replicate_keys_checked": True,
                         "scored_pair_sets_match": True, "panel_expected_847_of_864": True,
                         "no_model_services_or_downloads": True},
    }
    for name, rows in {
        "panel_pairs.csv": [panels[pid] for pid in ids], "panel_calls.csv": calls,
        "judge_pair_modals.csv": judge_rows, "panel_category_summary.csv": category_rows,
        "raw_call_L4_rates.csv": raw_rates, "system_calls.csv": system_calls,
        "system_call_L4_rates.csv": system_call_rates,
        "system_pair_predictions.csv": system_pairs, "system_pair_agreement.csv": agreements,
        "q6_matched_pairs.csv": q6_pairs,
        "q6_mcnemar.csv": [{"comparison": k, **{a: b for a, b in v.items() if a != "matched_pair_ids"}} for k, v in q6.items()],
        "q7_entropy_permutation.csv": [{"analysis": k, **{a: b for a, b in v.items() if a != "pair_ids"}} for k, v in q7.items()],
        "leave_one_judge_out.csv": lojo, "risk_recomputed.csv": risks,
        "leave_one_judge_out_pairs.csv": lojo_pair_summary,
        "S1_category_origin_risk.csv": risk_categories, "source_comparison_differences.csv": differences,
        "definition_side_origins.csv": [{"pair_id": p["pair_id"], "category": p["category"],
             "scored": p["pair_id"] in scored, "side": side, "has_definition_backfill": bool(p[side].get("definition_backfill")),
             "grounding": (p[side].get("definition_backfill") or {}).get("grounding", "no-backfill-record"),
             "confidence": (p[side].get("definition_backfill") or {}).get("confidence"),
             "definition_source": p[side].get("definition_source")} for p in benchmark.values() for side in ("construct_1", "construct_2")],
    }.items():
        write_csv(out / name, rows)
    write_json(out / "reanalysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "replication" / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=49999)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.bootstraps < 1 or args.permutations < 1:
        parser.error("resampling counts must be positive")
    result = run(args.data_dir, args.output_dir, args.bootstraps, args.permutations, args.seed)
    print(json.dumps({"panel_attempts": result["panel"]["n_attempts"], "panel_valid": result["panel"]["n_valid"],
                      "reliability": result["panel"]["reliability"], "Q6": result["Q6"]["comparisons"],
                      "risk": result["risk"]["current_summary"],
                      "source_comparison_differences": len(result["source_comparisons"]["differences"])}, indent=2))


if __name__ == "__main__":
    main()
