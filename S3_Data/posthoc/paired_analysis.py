"""Offline cross-check; no API or file mutation."""
from pathlib import Path
import csv,hashlib,json
s=Path(__file__).resolve().parents[1]
raw=(s/"analysis/per_cell.csv").read_bytes()
expected=json.loads((s/"posthoc/paired_diagnostics.json").read_text())
assert hashlib.sha256(raw).hexdigest()==expected["source_sha256"]
rows=list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
def m(r):return r["match"]=="True"
actual=[]
for p in ("glm","qwen"):
 for c in ("labels","definitions"):
  by={a:{(r["edge_id"],r["orientation"]):r for r in rows if (r["provider"],r["condition"],r["arm"])==(p,c,a)} for a in "DFL"}
  for comp in "LD":
   pp=[(by["F"][k],by[comp][k]) for k in by["F"]]
   def t(v):return {"n":len(v),"gains":sum(m(a) and not m(b) for a,b in v),"losses":sum(m(b) and not m(a) for a,b in v)}
   vv=[(a,b) for a,b in pp if a["valid_returned"]==b["valid_returned"]=="True"]
   z=t(pp)
   actual.append({"provider":p,"condition":c,"contrast":"F-"+comp,**z,"net":z["gains"]-z["losses"],
    "both_valid":t(vv),"changed_labels_both_valid":sum(a["predicted_label"]!=b["predicted_label"] for a,b in vv)})
assert actual==expected["contrasts"]
print("PASS: all eight paired contrasts reproduced from frozen CSV.")

