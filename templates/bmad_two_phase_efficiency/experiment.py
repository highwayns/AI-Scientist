
# templates/two_phase_efficiency/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass
rng = np.random.default_rng

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_path: str | None = None
    seed: int = 0
    treatments: tuple = ("two_phase","single_phase")
    # 机制/参数
    phase1_investment_ratio: float = 0.4
    review_quality: float = 0.8
    requirement_volatility: float = 0.3
    batch_size: int = 4
    parallelism: int = 4
    wip_limit: int = 4
    context_switch_cost: float = 1.1
    engineer_cost_rate: float = 1.0

def simulate_once(cfg: Config, treatment: str, r: np.random.Generator):
    n_items = 240
    # 需求到达与规模
    work_size = r.lognormal(mean=0.0, sigma=0.6, size=n_items)  # 功能点规模
    volatility = cfg.requirement_volatility

    # Phase-1投入与评审质量影响  → 早期发现比例、返工与缺陷
    p1 = cfg.phase1_investment_ratio
    rq = cfg.review_quality

    early_detection = np.clip(0.2 + 0.9*p1*rq*(1.0 - 0.5*volatility), 0.05, 0.95)
    # 单阶段则近似为较低早期发现率
    if treatment == "single_phase":
        early_detection *= 0.5

    # 返工/缺陷泄漏近似：受早期发现率与波动影响
    rework_rate = np.clip(0.35*(1.0 - early_detection) + 0.3*volatility, 0.02, 0.9)
    defect_leakage_ppk = np.clip(0.6*(1.0 - early_detection) + 0.4*volatility, 0.05, 1.5)

    # Review等待：受并行/在制、批量与评审质量影响
    base_review = r.lognormal(mean=-0.2, sigma=0.4, size=n_items)
    review_latency_hours = float(np.mean(base_review) * (1.0 + max(0, cfg.batch_size-3)*0.1) * (1.0 + max(0, cfg.parallelism-4)*0.05))

    # 开发/测试耗时（考虑返工与上下文切换）
    dev_time = work_size * (1.0 + rework_rate) * cfg.context_switch_cost
    test_time = work_size * (0.6 - 0.2*early_detection + 0.3*volatility)

    # 两阶段：前置评审开销带来额外时间，但减少返工
    if treatment == "two_phase":
        p1_overhead = work_size * p1 * (1.0 - 0.3*rq)  # 评审效率越高开销越低
        total_time = dev_time + test_time + p1_overhead + base_review
        test_pass_rate = float(0.9 - 0.2*volatility + 0.1*rq)
        change_failure_rate = float(np.clip(0.08 - 0.04*rq + 0.1*volatility, 0.02, 0.25))
    else:
        total_time = dev_time + test_time + base_review
        test_pass_rate = float(0.85 - 0.25*volatility)
        change_failure_rate = float(np.clip(0.12 + 0.15*volatility, 0.05, 0.4))

    # DORA近似
    lead_times = total_time / np.maximum(1, cfg.parallelism) + r.normal(0.0, 0.05, size=n_items)
    lead_time_median = float(np.median(lead_times))
    deploy_freq_per_week = float(7.0 / (np.mean(lead_times) + 1e-6))
    mttr_hours = float(np.clip(np.mean(lead_times) * (0.15 + 0.1*volatility), 0.5, 24.0))

    # 成本：人力成本随时间，早期评审降低返工成本；考虑批量带来的等待损失
    person_hours = float(np.mean(total_time) * (1.0 + 0.05*max(0, cfg.batch_size-4)))
    cost_per_feature_unit = float(cfg.engineer_cost_rate * person_hours)

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": change_failure_rate,
        "metric/mttr_hours": mttr_hours,
        "metric/rework_rate": float(np.mean(rework_rate)),
        "metric/defect_leakage_ppk": float(np.mean(defect_leakage_ppk)),
        "metric/review_latency_hours": review_latency_hours,
        "metric/cost_per_feature_unit": cost_per_feature_unit,
        "metric/test_pass_rate": test_pass_rate,
        "meta/mode": "simulation",
        "meta/treatment": treatment
    }

def run_observational(dataset_path: str):
    # 需要包含（示例）:
    # pr_open_ts, pr_merge_ts, is_failed_deploy, restore_hours,
    # test_pass_rate, rework_rate, defect_leakage_ppk, review_latency_hours,
    # cost_per_feature_unit
    df = pd.read_csv(dataset_path)
    lead_time = (pd.to_datetime(df["pr_merge_ts"]) - pd.to_datetime(df["pr_open_ts"])).dt.total_seconds() / 3600.0
    lead_time_median = float(lead_time.median())
    deploy_freq_per_week = float(df.shape[0] / max(1, (lead_time.sum()/24.0/7.0)))
    change_failure_rate = float(df["is_failed_deploy"].mean())
    mttr_hours = float(df["restore_hours"].replace(0, np.nan).median())

    test_pass_rate = float(df.get("test_pass_rate", pd.Series([np.nan]*len(df))).mean())
    rework_rate = float(df.get("rework_rate", pd.Series([np.nan]*len(df))).mean())
    defect_leakage_ppk = float(df.get("defect_leakage_ppk", pd.Series([np.nan]*len(df))).mean())
    review_latency_hours = float(df.get("review_latency_hours", pd.Series([np.nan]*len(df))).mean())
    cost_per_feature_unit = float(df.get("cost_per_feature_unit", pd.Series([np.nan]*len(df))).mean())

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": change_failure_rate,
        "metric/mttr_hours": mttr_hours,
        "metric/rework_rate": rework_rate,
        "metric/defect_leakage_ppk": defect_leakage_ppk,
        "metric/review_latency_hours": review_latency_hours,
        "metric/cost_per_feature_unit": cost_per_feature_unit,
        "metric/test_pass_rate": test_pass_rate,
        "meta/mode": "observational",
        "meta/treatment": "unknown"
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["simulation","observational"], default="simulation")
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase1_investment_ratio", type=float, default=0.4)
    ap.add_argument("--review_quality", type=float, default=0.8)
    ap.add_argument("--requirement_volatility", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--wip_limit", type=int, default=4)
    ap.add_argument("--context_switch_cost", type=float, default=1.1)
    ap.add_argument("--engineer_cost_rate", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode,
        dataset_path=args.dataset_path,
        seed=args.seed,
        phase1_investment_ratio=args.phase1_investment_ratio,
        review_quality=args.review_quality,
        requirement_volatility=args.requirement_volatility,
        batch_size=args.batch_size,
        parallelism=args.parallelism,
        wip_limit=args.wip_limit,
        context_switch_cost=args.context_switch_cost,
        engineer_cost_rate=args.engineer_cost_rate
    )

    r = rng(cfg.seed)
    rows = []

    if cfg.mode == "observational" and cfg.dataset_path:
        res = run_observational(cfg.dataset_path)
        rows.append(res)
    else:
        for treatment in cfg.treatments:
            res = simulate_once(cfg, treatment, r)
            rows.append(res)

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
