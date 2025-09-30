
# templates/model_drift_xai/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Tuple

rng = np.random.default_rng

def ece(prob, y, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(prob, bins) - 1
    e = 0.0
    for b in range(n_bins):
        sel = (idx==b)
        if sel.sum()==0: 
            continue
        conf = prob[sel].mean()
        acc = y[sel].mean()
        e += abs(acc - conf) * sel.mean()
    return float(e)

def ks_min_p(pre, post):
    from scipy.stats import ks_2samp
    pvals = []
    for j in range(pre.shape[1]):
        p = ks_2samp(pre[:,j], post[:,j]).pvalue
        pvals.append(p)
    return float(np.min(pvals))

def psi(pre, post, bins=10):
    brks = np.quantile(pre, np.linspace(0,1,bins+1))
    def hist(a):
        return np.histogram(a, bins=brks)[0] + 1e-6
    p = hist(pre) ; q = hist(post)
    p = p/p.sum(); q=q/q.sum()
    return float(np.sum((q-p)*np.log(q/p)))

def kl_div(p, q, eps=1e-6):
    p = p + eps; q = q + eps
    p/=p.sum(); q/=q.sum()
    return float(np.sum(p*np.log(p/q)))

def topk_consistency(w1, w2, k=5):
    r1 = np.argsort(-np.abs(w1))[:k]
    r2 = np.argsort(-np.abs(w2))[:k]
    return float(len(set(r1)&set(r2))/k)

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_ref: str | None = None
    seed: int = 0
    treatments: tuple = ("joint_xai","baseline_drift")
    drift_type: str = "covariate"
    drift_magnitude: float = 0.4
    window_size: int = 1000
    alert_threshold: float = 0.1
    use_shap_proxy: bool = True

def gen_stream(cfg: Config, r: np.random.Generator):
    n = cfg.window_size
    d = 8
    X_pre = r.normal(0,1,(n,d))
    w_true = r.normal(0,1,d)
    logits_pre = X_pre @ w_true
    y_pre = (logits_pre + r.normal(0,1,n) > 0).astype(int)
    p_pre = 1/(1+np.exp(-logits_pre))

    X_post = X_pre.copy()
    if cfg.drift_type in ["covariate","mixed"]:
        X_post += cfg.drift_magnitude * r.normal(0,1,(n,d))
    if cfg.drift_type in ["label","concept","mixed"]:
        w2 = w_true + cfg.drift_magnitude * r.normal(0,0.5,d)
    else:
        w2 = w_true
    logits_post = X_post @ w2
    y_post = (logits_post + r.normal(0,1,n) > 0).astype(int)
    p_post = 1/(1+np.exp(-logits_post))
    return X_pre, y_pre, p_pre, X_post, y_post, p_post, w_true, w2

def run_once(cfg: Config, treatment: str, r: np.random.Generator):
    X_pre, y_pre, p_pre, X_post, y_post, p_post, w1, w2 = gen_stream(cfg, r)

    from sklearn.metrics import roc_auc_score
    au_pre = float(roc_auc_score(y_pre, p_pre))
    au_post = float(roc_auc_score(y_post, p_post))
    ece_pre = ece(p_pre, y_pre)
    ece_post = ece(p_post, y_post)

    psi_vals = [psi(X_pre[:,j], X_post[:,j]) for j in range(X_pre.shape[1])]
    kl_vals = [kl_div(np.histogram(X_pre[:,j], bins=20)[0].astype(float),
                      np.histogram(X_post[:,j], bins=20)[0].astype(float)) for j in range(X_pre.shape[1])]
    from scipy.stats import ks_2samp
    ks_p = min(ks_2samp(X_pre[:,j], X_post[:,j]).pvalue for j in range(X_pre.shape[1]))
    PSI = float(np.mean(psi_vals))
    KL = float(np.mean(kl_vals))

    xai_k = 5
    xai_cons = topk_consistency(w1, w2, k=xai_k)
    attr_shift = float(np.linalg.norm(w1 - w2, ord=1)/len(w1))

    if treatment == "joint_xai":
        drift_score = (1-np.clip(ks_p,0,1)) * 0.5 + min(1.0, abs(au_pre-au_post))*0.2 + (1-xai_cons)*0.3
    else:
        drift_score = (1-np.clip(ks_p,0,1)) * 0.7 + min(1.0, abs(au_pre-au_post))*0.3

    alerts = int(drift_score > cfg.alert_threshold)
    true_drift = cfg.drift_magnitude > 0.2
    tp = 1 if (alerts and true_drift) else 0
    fp = 1 if (alerts and not true_drift) else 0
    fn = 1 if ((not alerts) and true_drift) else 0
    precision = float(tp / (tp+fp)) if (tp+fp)>0 else 0.0
    recall = float(tp / (tp+fn)) if (tp+fn)>0 else 0.0
    avg_alerts_per_day = float(alerts)

    fn_rate = float(((p_post>0.5)&(y_post==1)).mean()==0)
    fn_cost = float(fn * max(0.1, 0.5*cfg.drift_magnitude + 0.2*fn_rate))

    return {
        "metric/auroc_pre": au_pre,
        "metric/auroc_post": au_post,
        "metric/calib_ece_pre": ece_pre,
        "metric/calib_ece_post": ece_post,
        "metric/psi": PSI,
        "metric/kl_div": KL,
        "metric/ks_p_min": float(ks_p),
        "metric/xai_consistency": xai_cons,
        "metric/attr_shift": attr_shift,
        "metric/alert_precision": precision,
        "metric/alert_recall": recall,
        "metric/avg_alerts_per_day": avg_alerts_per_day,
        "metric/fn_cost": fn_cost,
        "meta/mode": "simulation",
        "meta/treatment": treatment,
        "meta/drift_type": cfg.drift_type,
        "meta/drift_magnitude": cfg.drift_magnitude
    }

def run_observational(prefix: str):
    pre = pd.read_csv(os.path.join(prefix, "pre.csv"))
    post = pd.read_csv(os.path.join(prefix, "post.csv"))

    y_pre = pre["y"].values.astype(int); p_pre = pre["pred_prob"].values.astype(float)
    y_post = post["y"].values.astype(int); p_post = post["pred_prob"].values.astype(float)
    X_pre = pre[[c for c in pre.columns if c not in ["y","pred_prob"]]].values
    X_post = post[[c for c in post.columns if c not in ["y","pred_prob"]]].values

    from sklearn.metrics import roc_auc_score
    au_pre = float(roc_auc_score(y_pre, p_pre))
    au_post = float(roc_auc_score(y_post, p_post))
    ece_pre = ece(p_pre, y_pre)
    ece_post = ece(p_post, y_post)

    psi_vals = [psi(X_pre[:,j], X_post[:,j]) for j in range(X_pre.shape[1])]
    kl_vals = [kl_div(np.histogram(X_pre[:,j], bins=20)[0].astype(float),
                      np.histogram(X_post[:,j], bins=20)[0].astype(float)) for j in range(X_pre.shape[1])]
    from scipy.stats import ks_2samp
    ks_p = min(ks_2samp(X_pre[:,j], X_post[:,j]).pvalue for j in range(X_pre.shape[1]))
    PSI = float(np.mean(psi_vals))
    KL = float(np.mean(kl_vals))

    w1 = X_pre.mean(0); w2 = X_post.mean(0)
    xai_cons = float(max(0.0, 1.0 - np.linalg.norm(w1-w2)/ (np.linalg.norm(abs(w1))+1e-6)))
    attr_shift = float(np.linalg.norm(w1-w2, ord=1)/len(w1))

    precision=recall=avg_alerts=fn_cost=0.0
    return {
        "metric/auroc_pre": au_pre,
        "metric/auroc_post": au_post,
        "metric/calib_ece_pre": ece_pre,
        "metric/calib_ece_post": ece_post,
        "metric/psi": PSI,
        "metric/kl_div": KL,
        "metric/ks_p_min": float(ks_p),
        "metric/xai_consistency": xai_cons,
        "metric/attr_shift": attr_shift,
        "metric/alert_precision": precision,
        "metric/alert_recall": recall,
        "metric/avg_alerts_per_day": avg_alerts,
        "metric/fn_cost": fn_cost,
        "meta/mode": "observational",
        "meta/treatment": "unknown",
        "meta/drift_type": "unknown",
        "meta/drift_magnitude": float("nan")
    }

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["simulation","observational"], default="simulation")
    ap.add_argument("--dataset_ref", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift_type", choices=["covariate","label","concept","mixed"], default="covariate")
    ap.add_argument("--drift_magnitude", type=float, default=0.4)
    ap.add_argument("--window_size", type=int, default=1000)
    ap.add_argument("--alert_threshold", type=float, default=0.1)
    ap.add_argument("--use_shap_proxy", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode,
        dataset_ref=args.dataset_ref,
        seed=args.seed,
        drift_type=args.drift_type,
        drift_magnitude=args.drift_magnitude,
        window_size=args.window_size,
        alert_threshold=args.alert_threshold,
        use_shap_proxy=args.use_shap_proxy
    )

    r = rng(cfg.seed)
    rows = []

    if cfg.mode == "observational" and cfg.dataset_ref:
        res = run_observational(cfg.dataset_ref)
        rows.append(res)
    else:
        for treatment in cfg.treatments:
            res = run_once(cfg, treatment, r)
            rows.append(res)

    series = np.array([[row[k] for k in [
        "metric/auroc_pre","metric/auroc_post",
        "metric/calib_ece_pre","metric/calib_ece_post"
    ]] for row in rows], dtype=float)
    np.save(os.path.join(args.out_dir, "series.npy"), series)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "per_run_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({k: rows[0][k] for k in rows[0].keys()}, f, indent=2)
