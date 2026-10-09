"""Standalone numerical validation; reads results without overwriting experiments.

Uses an independent weight-form history enumerator and chronological recurrence.
Requires only Python and NumPy. No DOCX, render directory, or desktop dependency.
"""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import comparison_experiments as experiment
import calibration_experiments as calibration

BASE = Path(__file__).resolve().parent
start = time.perf_counter()
R = json.loads((BASE / "comparison_results.json").read_text())
C = json.loads((BASE / "calibration_results.json").read_text())
P = json.loads((BASE.parent / "paper2.json").read_text(encoding="utf-8"))
checks = []

def checked(name, condition, detail=None):
    assert condition, name
    checks.append({"check": name, "pass": True, "detail": detail})

def close(a, b, atol=1e-9, rtol=1e-9):
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))

def weight_probability(beta, alpha, one, zero):
    w1 = math.exp(beta) * np.power(1 + one, alpha)
    w0 = np.power(1 + zero, alpha)
    return w1 / (w0 + w1)

def enumerate_independently(alpha, separation, length):
    # Each row is a complete binary history; preceding counts exclude current A.
    bits = ((np.arange(2**length)[:, None] >> np.arange(length)) & 1).astype(bool)
    n1 = np.cumsum(bits, axis=1) - bits
    n0 = np.arange(length)[None, :] - n1
    pm = weight_probability(-separation, alpha, n1, n0)
    pp = weight_probability(separation, alpha, n1, n0)
    qm = np.prod(np.where(bits, pm, 1-pm), axis=1)
    qp = np.prod(np.where(bits, pp, 1-pp), axis=1)
    bm = np.sum(np.minimum(pm, 1-pm), axis=1)
    bp = np.sum(np.minimum(pp, 1-pp), axis=1)
    im = np.sum(pm*(1-pm), axis=1)
    ip = np.sum(pp*(1-pp), axis=1)
    return {"mass": [qm.sum(), qp.sum()], "exact_bayes_risk": .5*np.minimum(qm,qp).sum(),
            "affinity": np.sqrt(qm*qp).sum(), "expected_budget": .5*(qm@bm+qp@bp),
            "expected_information": .5*(qm@im+qp@ip), "expected_budget_minus": qm@bm,
            "kl_minus_plus": qm@np.log(qm/qp)}

checked("Recorded configuration counts", len(R["core"])==18 and len(R["stress"])==4 and len(R["exact_short_markets"])==54 and len(C["cases"])==32)
for x in R["exact_short_markets"]:
    independent = enumerate_independently(x["alpha"], x["separation"], x["length"])
    checked(f"Independent complete-history enumeration α={x['alpha']}, d={x['separation']}, L={x['length']}",
            close(independent["mass"], [1,1]) and all(close(independent[k],x[k]) for k in ("exact_bayes_risk","affinity","expected_budget","expected_information")))
    checked(f"Finite divergence and affinity bounds α={x['alpha']}, d={x['separation']}, L={x['length']}",
            independent["kl_minus_plus"] <= .5*math.exp(2*x["separation"])*(2*x["separation"])**2*independent["expected_budget_minus"]+1e-10
            and x["exact_bayes_risk"] <= .5*x["affinity"]+1e-12 and x["expected_information"]<=x["expected_budget"]+1e-12)

benchmark = next(x for x in R["exact_short_markets"] if x["alpha"]==2 and x["separation"]==.5 and x["length"]==16)
checked("Independent earlier length-sixteen benchmark", close(benchmark["exact_bayes_risk"], .2667762473576409, 1e-13), benchmark)

def independent_chronological_check(reset=None, retention=0, response=False):
    n=7
    seed=20261071
    rng=np.random.default_rng(seed)
    audit_rng=np.random.default_rng(seed+900000)
    truth=np.repeat([-.5,.5],n)
    one=np.zeros(2*n); zero=np.zeros(2*n)
    b=np.zeros(2*n); info=np.zeros(2*n); ratio=np.zeros(2*n)
    ia=np.zeros(2*n); aa=np.zeros(2*n)
    result=experiment.simulate_pair(2,.5,reset,retention,response,seed,n)
    for t in range(max(experiment.HORIZONS)):
        if reset and t%reset==0:
            one*=retention; zero*=retention
        pt=np.array([weight_probability(v,2,one[j],zero[j]) for j,v in enumerate(truth)])
        pm=weight_probability(-.5,2,one,zero); pp=weight_probability(.5,2,one,zero)
        logratio=np.log1p(one)-np.log1p(zero)
        v=pt*(1-pt)
        b+=np.minimum(pt,1-pt); info+=v; ia+=v*logratio; aa+=v*logratio**2
        y=rng.random(2*n)<pt
        ratio+=np.log(np.where(y,pp,1-pp)/np.where(y,pm,1-pm))
        one+=y; zero+=1-y
        if response:
            qt=np.array([weight_probability(v,1,0,0) for v in truth])
            z=audit_rng.random(2*n)<qt
            qm=weight_probability(-.5,1,0,0); qp=weight_probability(.5,1,0,0)
            ratio+=np.log(np.where(z,qp,1-qp)/np.where(z,qm,1-qm))
            info+=qt*(1-qt)
        if t+1 in experiment.HORIZONS:
            j=experiment.HORIZONS.index(t+1); row=result[j]
            error=np.where(truth<0,ratio>0,ratio<0).astype(float)
            error[np.abs(ratio)<=1e-12]=.5
            gap=np.abs(weight_probability(.5,2,one,zero)-weight_probability(-.5,2,one,zero))
            checked(f"Independent pre-allocation recurrence reset={reset}, ρ={retention}, response={response}, T={t+1}",
                    all(close(row[k],v,1e-7) for k,v in [("budget",b),("information",info),("error",error),("forecast_gap",gap),("information_beta_alpha",ia),("information_alpha_alpha",aa)]))
            if reset and retention<1:
                floor=1/(1+math.exp(.5)*(1+reset/(1-retention))**2)
                checked(f"Retained-state residual floor ρ={retention}, T={t+1}", np.all(b>=floor*(t+1)-1e-10))

for args in [(None,0,False),(16,0,False),(16,.5,False),(16,.9,False),(16,1,False),(None,0,True)]:
    independent_chronological_check(*args)

single=next(x for x in R["core"] if x["alpha"]==2 and x["separation"]==.5 and x["reset_length"] is None)
retained=next(x for x in R["stress"] if x["retention"]==1)
audited=next(x for x in R["stress"] if x["response"])
audit_variance=weight_probability(.5,1,0,0)*(1-weight_probability(.5,1,0,0))
for s,u,a in zip(single["rows"],retained["rows"],audited["rows"]):
    checked(f"Stored matched-recipient and audit identities T={s['observations']}",
            s["bayes_risk"]==u["bayes_risk"] and s["budget"]==u["budget"] and s["budget"]==a["budget"]
            and close(a["information_beta_beta"]["mean"]-s["information_beta_beta"]["mean"],s["observations"]*audit_variance,1e-8))
for x in R["core"]+R["stress"]:
    for row in x["rows"]:
        checked(f"Risk weighting and scalar information bound α={x['alpha']}, d={x['separation']}, reset={x['reset_length']}, ρ={x['retention']}, response={x['response']}, T={row['observations']}",
                close(row["bayes_risk"],.5*sum(row["errors_by_law"])) and
                (x["response"] or row["information_beta_beta"]["mean"]<=row["budget"]["mean"]+1e-9))

# Rank discreteness and conservative ties, including a deterministic all-tie law.
checked("Rank ties included and resolution respected",
        close(calibration.rank_pvalues(np.array([0.,0.,2.]),np.array([-1.,0.,1.,2.,3.])),[1.,1.,.5,.5,.25])
        and calibration.rank_pvalues(np.zeros(999),np.array([0.]))[0]==1
        and calibration.rank_pvalues(np.zeros(999),np.array([1.]))[0]==.001)
with np.load(BASE/"calibration_banks.npz") as banks:
    checked("Complete ordered calibration banks",len(banks.files)==120 and all(len(banks[k])==999 and np.all(np.diff(banks[k])>=0) for k in banks.files))
for x in C["cases"]:
    baseline=next(v for v in C["cases"] if v["true_alpha"]==x["true_alpha"] and v["reset_length"]==x["reset_length"] and v["procedure"]=="baseline_nuisance_union")
    expanded=next(v for v in C["cases"] if v["true_alpha"]==x["true_alpha"] and v["reset_length"]==x["reset_length"] and v["procedure"]=="expanded_nuisance_union")
    checked(f"Calibration pooling and nuisance nesting α={x['true_alpha']}, reset={x['reset_length']}, procedure={x['procedure']}",
            close(x["coverage"],np.mean(x["coverage_by_seed_bank"])) and expanded["coverage"]>=baseline["coverage"] and expanded["mean_grid_set_cardinality"]>=baseline["mean_grid_set_cardinality"])

tables=[p["table"] for s in P["sections"] for p in s["paragraphs"] if isinstance(p,dict) and "table" in p]
checked("Manuscript table count and row counts", [len(t["rows"]) for t in tables]==[6,9,6,16])
for row,x in zip(tables[0]["rows"],[x for x in R["exact_short_markets"] if x["alpha"]==2 and x["separation"]==.5]):
    checked(f"Exact table numerical payload L={x['length']}",row==[x["length"],f"{100*x['exact_bayes_risk']:.4f}",f"{x['affinity']:.6f}",f"{x['expected_budget']:.6f}",f"{x['expected_information']:.6f}"])
for row in tables[1]["rows"]:
    alpha=row[0]; d=float(row[1][1:])
    s=next(x for x in R["core"] if x["alpha"]==alpha and x["separation"]==d and x["reset_length"] is None)
    r=next(x for x in R["core"] if x["alpha"]==alpha and x["separation"]==d and x["reset_length"]==16)
    rows=[s["rows"][1],s["rows"][-1],r["rows"][1],r["rows"][-1]]
    formatted=["0*" if z["bayes_risk"]==0 else f"{100*z['bayes_risk']:.2f} ({100*z['risk_mc_se']:.3f})" for z in rows]
    checked(f"Classification table numerical payload α={alpha}, d={d}",row[2:]==formatted)
stress_conditions=[single,
    next(x for x in R["core"] if x["alpha"]==2 and x["separation"]==.5 and x["reset_length"]==16),
    *[next(x for x in R["stress"] if x["retention"]==rho) for rho in (.5,.9,1.)], audited]
for table_row, config in zip(tables[2]["rows"],stress_conditions):
    z=config["rows"][-1]
    risk=f"{100*z['bayes_risk']:.2f} ({100*z['risk_mc_se']:.3f})" if z["bayes_risk"] else f"0 observed; upper {100*z['risk_mc_95_interval'][1]:.4f}"
    expected=[risk,f"{z['budget']['mean']:.3f} ({z['budget']['mc_se']:.3f})",f"{z['information_beta_beta']['mean']:.3f} ({z['information_beta_beta']['mc_se']:.3f})"]
    checked(f"Response/persistence table numerical payload {table_row[0]}", table_row[1:]==expected)
for row in tables[3]["rows"]:
    label,a,short,cov,size=row
    procedure={"Known α":"correct_known_alpha","Fix α=2":"wrong_fixed_alpha2","Union {.5,1,2}":"baseline_nuisance_union","Union {.5,1,1.5,2}":"expanded_nuisance_union"}[short]
    x=next(v for v in C["cases"] if v["true_alpha"]==a and v["procedure"]==procedure and v["reset_length"]==(16 if label=="Reset" else None))
    checked(f"Calibration table numerical payload {label}, α={a}, {short}",cov==f"{100*x['coverage']:.2f} ({100*x['coverage_mc_se_conditioning_on_calibration_banks']:.3f})" and size==f"{x['mean_grid_set_cardinality']:.3f}")
checked("Article and abstract length",6000<=P["word_count"]<=8000 and 150<=len(P["abstract"].split())<=250)

out={"status":"pass","checks_count":len(checks),"checks":checks,"runtime_seconds":time.perf_counter()-start,
     "inputs_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/"comparison_results.json",BASE/"calibration_results.json",BASE/"calibration_banks.npz",BASE.parent/"paper2.json"]},
     "limitations":"Numerical validation of the implemented finite-horizon synthetic experiment; no field validation or infinite-horizon conclusion."}
(BASE/"validation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(f"Scientific validation passed: {len(checks)} checks in {out['runtime_seconds']:.2f}s.")
