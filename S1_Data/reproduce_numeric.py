"""Offline reproduction from numeric_inputs.json; no manuscript, summaries, network or models read.

Run: python reproduce_numeric.py --input numeric_inputs.json --output numeric_reproduced.json
Dependency: PyYAML==6.0.1, because the reused arithmetic module imports its raw-file loader.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import statistics
import reanalyze_frozen as core

def reproduce(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    bench = {p["pair_id"]:p for p in d["benchmark"]}
    scored = {pid:p for pid,p in bench.items() if p["category"] != "HN" and not p.get("unscored")}
    judges = [j["alias_label"] for j in d["panel_metadata"]["judges"]]
    panel = {p["pair_id"]:core.aggregate(p["panel_judgments"],judges) for p in d["panel"]}
    assert len(bench)==100 and len(scored)==len(panel)==96
    ids = sorted(panel)
    rating = [core.counts_row(panel[pid]["per_judge_modal"].values()) for pid in ids]
    reliability = {"fleiss_kappa":core.fleiss_kappa([r for r in rating if sum(r)==3]),
                   "krippendorff_alpha_nominal":core.krippendorff_alpha(rating),
                   "bootstrap":core.bootstrap_reliability(rating,3,5000,20260914)}
    categories = ["ALL",*core.CATEGORIES,"NONCONTROLS"]
    system_agreement=[]
    system_maps={}
    calibration={}
    for name, records in d["systems"].items():
        bypair={pid:[] for pid in ids}
        if name != "S6_llm_direct":
            thresholds, meta=core.calibrate(records,bench)
            calibration[name]={"thresholds":thresholds,**meta}
        for r in records:
            if r["error"] is None:
                label=r["predicted_label"] if name=="S6_llm_direct" else core.normalize(r["raw_score"],bench[r["pair_id"]],thresholds)
                if label in core.LABELS: bypair[r["pair_id"]].append(label)
        predictions={pid:core.mode(vals)[0] for pid,vals in bypair.items()}
        system_maps[name]=predictions
        for cat in categories:
            subset=[pid for pid in ids if panel[pid]["majority_label"] in core.LABELS and
                    (cat=="ALL" or cat=="NONCONTROLS" and bench[pid]["category"]!="D" or bench[pid]["category"]==cat)]
            matches=sum(predictions[pid]==panel[pid]["majority_label"] for pid in subset)
            system_agreement.append({"system":name,"category":cat,"n":len(subset),"matches":matches,"agreement":matches/len(subset)})
    q6={}
    l3=[pid for pid in ids if panel[pid]["majority_label"]=="L3"]
    a=[system_maps["S6_llm_direct"][pid]=="L3" for pid in l3]
    for name in ("S2_wulff_mata","S5_generic_embed"):
        q6[name]=core.mcnemar(a,[system_maps[name][pid]=="L3" for pid in l3])
    adjusted=core.holm({k:v["p_exact_two_sided"] for k,v in q6.items()})
    for k,v in adjusted.items():q6[k]["holm_adjusted_p"]=v
    q7={}
    entropy_means={}
    for variant in ("raw_calls","judge_modals"):
        ent={pid:panel[pid]["vote_entropy"] if variant=="raw_calls" else core.entropy(panel[pid]["cross_judge_modal_distribution"]) for pid in ids}
        entropy_means[variant]={cat:statistics.mean(ent[pid] for pid in ids if bench[pid]["category"]==cat) for cat in core.CATEGORIES}
        for subset in ("all","noncontrols"):
            chosen=[pid for pid in ids if subset=="all" or bench[pid]["category"]!="D"]
            q7[f"{variant}_{subset}"]=core.permutation_difference([ent[pid] for pid in chosen],[bench[pid]["category"]=="A_prime" for pid in chosen],49999,20260914)
    risks=[core.risk_row(scored[pid],panel[pid]) for pid in ids]
    omissions=Counter()
    stable=0
    for pid in ids:
        if panel[pid]["majority_label"] not in core.LABELS:continue
        states=[]
        for omitted in judges:
            remaining=[panel[pid]["per_judge_modal"][j] for j in judges if j!=omitted]
            state="insufficient" if any(v is None for v in remaining) else "became_tie" if len(set(remaining))>1 else "unchanged"
            omissions[state]+=1;states.append(state)
        stable+=all(s=="unchanged" for s in states)
    result={"n_attempts":sum(len(p["panel_judgments"]) for p in d["panel"]),
            "n_valid":sum(p["n_valid"] for p in panel.values()),
            "panel_label_counts":dict(Counter(p["majority_label"] for p in panel.values())),
            "reliability":reliability,"system_pair_agreement":system_agreement,"calibration":calibration,
            "entropy_category_means":entropy_means,"Q6":q6,"Q7":q7,
            "origin_counts":core.origin_summary(list(scored.values())),
            "risk_priority_counts":dict(Counter(r["human_validation_priority"] for r in risks)),
            "operational_tier":sum(r["gold_tier"] for r in risks),"strict_tier":sum(r["gold_tier_strict"] for r in risks),
            "leave_one_judge_out_resolved_runs":dict(omissions),"stable_all_omissions_pairs":stable}
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path(__file__).with_name("numeric_inputs.json"))
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("numeric_reproduced.json"))
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve(): raise ValueError("Refusing to overwrite numeric inputs")
    result=reproduce(args.input)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(args.output)
