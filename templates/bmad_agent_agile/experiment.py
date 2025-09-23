
# templates/bmad_agent_agile/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass
rng = np.random.default_rng

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_path: str | None = None
    sim_rounds: int = 2000
    seed: int = 0
    treatments: tuple = ("bmad_agent", "silo_agile")
    # flow/behavior params
    wip_limit: int = 3
    review_capacity: int = 4
    handoff_cost: float = 1.2
    agent_orchestration_eff: float = 1.1
    role_entropy_boost: float = 0.2

def simulate_once(cfg: Config, treatment: str, r: np.random.Generator):
    # --- toy flow model ---
    n_items = 200
    dev_time = r.lognormal(mean=0.0, sigma=0.4, size=n_items)  # base work
    review_time = r.lognormal(mean=-0.1, sigma=0.3, size=n_items)

    # treatment effects
    if treatment == "bmad_agent":
        # fewer handoffs + better coordination
        dev_time *= 1.0 / cfg.agent_orchestration_eff
        review_time *= 1.0 / (cfg.agent_orchestration_eff * 0.95)
        cross_role_review_ratio = 0.55 + cfg.role_entropy_boost  # more cross-discipline
        role_entropy = 1.1 + cfg.role_entropy_boost
        change_failure_rate = max(0.05, 0.12 - 0.03*cfg.agent_orchestration_eff)
    else:
        dev_time *= cfg.handoff_cost
        review_time *= 1.05 * cfg.handoff_cost
        cross_role_review_ratio = 0.35
        role_entropy = 0.9
        change_failure_rate = 0.12

    # WIP-constrained flow approximation
    q = 0.0
    lead_times = []
    wip_samples = []
    queue_share = []

    for t_dev, t_rev in zip(dev_time, review_time):
        # queue wait under WIP
        wait = max(0.0, (q - cfg.wip_limit) * 0.2)
        total = wait + t_dev + (t_rev / max(1, cfg.review_capacity/4))
        lead_times.append(total)
        q = max(0.0, q + t_dev - 0.8)  # drain factor
        wip_samples.append(min(q, cfg.wip_limit + 2))
        queue_share.append(wait / total if total > 0 else 0.0)

    lead_time_median = float(np.median(lead_times))
    deploy_freq_per_week = float(7.0 / (np.mean(lead_times) + 1e-6))
    mttr_hours = float(np.clip(np.mean(lead_times) * 0.2, 0.5, 24.0))

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": change_failure_rate,
        "metric/mttr_hours": mttr_hours,
        "metric/cross_role_review_ratio": float(cross_role_review_ratio),
        "metric/role_entropy": float(role_entropy),
        "metric/wip_avg": float(np.mean(wip_samples)),
        "metric/queue_time_share": float(np.mean(queue_share)),
        "meta/mode": "simulation",
        "meta/treatment": treatment
    }

def run_observational(dataset_path: str):
    # expect a CSV with columns:
    # pr_open_ts, pr_merge_ts, is_failed_deploy, restore_hours, reviewer_role, author_role, files_roles[]
    df = pd.read_csv(dataset_path)
    lead_time = (pd.to_datetime(df["pr_merge_ts"]) - pd.to_datetime(df["pr_open_ts"])).dt.total_seconds() / 3600.0
    lead_time_median = float(lead_time.median())
    deploy_freq_per_week = float(df.shape[0] / max(1, (lead_time.sum()/24.0/7.0)))
    change_failure_rate = float(df["is_failed_deploy"].mean())
    mttr_hours = float(df["restore_hours"].replace(0, np.nan).median())

    # cross-role review ratio
    cross = (df["reviewer_role"] != df["author_role"]).mean()
    # role entropy (approx): if files_roles is semi-colon list
    def entropy(roles):
        parts = str(roles).split(";")
        vals, cnts = np.unique(parts, return_counts=True)
        p = cnts / cnts.sum()
        return float(-(p * np.log(p + 1e-9)).sum())
    role_entropy = float(df["files_roles"].apply(entropy).mean())

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": change_failure_rate,
        "metric/mttr_hours": mttr_hours,
        "metric/cross_role_review_ratio": float(cross),
        "metric/role_entropy": float(role_entropy),
        "metric/wip_avg": float("nan"),
        "metric/queue_time_share": float("nan"),
        "meta/mode": "observational",
        "meta/treatment": "unknown"
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["simulation","observational"], default="simulation")
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim_rounds", type=int, default=2000)
    ap.add_argument("--wip_limit", type=int, default=3)
    ap.add_argument("--review_capacity", type=int, default=4)
    ap.add_argument("--handoff_cost", type=float, default=1.2)
    ap.add_argument("--agent_orchestration_eff", type=float, default=1.1)
    ap.add_argument("--role_entropy_boost", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode,
        dataset_path=args.dataset_path,
        sim_rounds=args.sim_rounds,
        seed=args.seed,
        wip_limit=args.wip_limit,
        review_capacity=args.review_capacity,
        handoff_cost=args.handoff_cost,
        agent_orchestration_eff=args.agent_orchestration_eff,
        role_entropy_boost=args.role_entropy_boost
    )

    rows = []
    r = rng(cfg.seed)

    if cfg.mode == "observational" and cfg.dataset_path:
        res = run_observational(cfg.dataset_path)
        rows.append(res)
    else:
        for treatment in cfg.treatments:
            res = simulate_once(cfg, treatment, r)
            rows.append(res)

    # persist
    import numpy as np
    series = np.array([[row[k] for k in [
        "metric/lead_time_median","metric/deploy_freq_per_week",
        "metric/change_failure_rate","metric/mttr_hours"
    ]] for row in rows], dtype=float)
    np.save(os.path.join(args.out_dir, "series.npy"), series)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "per_run_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({k: rows[0][k] for k in rows[0].keys()}, f, indent=2)
