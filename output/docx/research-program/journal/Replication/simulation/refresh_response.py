"""Refresh the response-channel stress with independently seeded response draws.

The complete main runner yields the same response results; this helper avoids
rerunning unchanged core experiments during development.
"""
from pathlib import Path
import json
import comparison_experiments as c

path = Path(__file__).resolve().parent / "comparison_results.json"
data = json.loads(path.read_text(encoding="utf-8"))
config = {"alpha": 2.0, "separation": 0.5, "reset_length": None, "retention": 0.0, "response": True}
runs = [c.simulate_pair(**config, seed=s) for s in c.SEEDS]
fresh = c.pool_rows(runs, config)
index = next(i for i, x in enumerate(data["stress"]) if x["response"])
data["stress"][index] = fresh
single = next(x for x in data["core"] if x["alpha"] == 2 and x["separation"] == .5 and x["reset_length"] is None)
for a, b in zip(single["rows"], fresh["rows"]):
    assert a["budget"] == b["budget"]
data["response_channel_rng"] = "Independent response stream seed = allocation seed + 900000; allocation paths exactly shared with no-response core."
path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(json.dumps(fresh, indent=2))
