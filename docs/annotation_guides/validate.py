"""python validate.py <kind> <input.jsonl> <output.jsonl>   kind: protect | probe | seg | minpairs"""
import json
import sys


def rd(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


kind, inp, out = sys.argv[1:4]
a, b = rd(inp), rd(out)
errors = []
key = "base" if kind == "minpairs" else "id"
if [x[key] for x in a] != [y.get(key) for y in b]:
    errors.append(f"ids/order differ: {len(a)} input rows, {len(b)} output rows")
for x, y in zip(a, b):
    if kind == "protect":
        g = y.get("gold_tiers")
        if not isinstance(g, dict):
            errors.append(f"{x['id']}: gold_tiers missing")
            continue
        n = len(x["words"])
        for k, v in g.items():
            if not k.isdigit() or int(k) >= n or v not in ("T1", "T2", "T3"):
                errors.append(f"{x['id']}: bad entry {k}:{v}")
            elif x["words"][int(k)][1] and all(not c.isalnum() for c in x["words"][int(k)][1]):
                errors.append(f"{x['id']}: punctuation labelled at {k}")
    elif kind == "probe":
        if not isinstance(y.get("gold_negation"), bool):
            errors.append(f"{x['id']}: gold_negation must be true/false")
    elif kind == "seg":
        gw = y.get("gold_words")
        if not isinstance(gw, list) or [s for w in gw for s in w.split("_")] != x["syllables"]:
            errors.append(f"{x['id']}: gold_words do not reproduce the syllables exactly")
    else:
        if not isinstance(y.get("keep"), bool) or not y.get("hypothesis_contradiction"):
            errors.append(f"{x['base']}: keep/hypothesis_contradiction missing")
        elif (not y["keep"] or y["hypothesis_contradiction"] != x["hypothesis_contradiction"]) and not y.get("note"):
            errors.append(f"{x['base']}: note required")
print("OK" if not errors else "\n".join(errors[:40]) + f"\n{len(errors)} error(s)")
