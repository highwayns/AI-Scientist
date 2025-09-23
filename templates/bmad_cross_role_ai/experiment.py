
# templates/bmad_cross_role_ai/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass
rng = np.random.default_rng

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_path: str | None = None
    seed: int = 0
    treatments: tuple = ("bmad_ai", "silo_agile")
    # flow/behavior params
    wip_limit: int = 3
    review_capacity: int = 4
    handoff_cost: float = 1.2
    ai_assist_coverage: float = 0.6
    ai_suggestion_accept_rate: float = 0.6
    quality_gate_threshold: float = 0.7
    orchestration_efficiency: float = 1.1

def simulate_once(cfg: Config, treatment: str, r: np.random.Generator):
    # --- toy flow model ---
    n_items = 240
    # 基础工作量（对数正态）
    dev_time = r.lognormal(mean=0.0, sigma=0.4, size=n_items)
    review_time = r.lognormal(mean=-0.1, sigma=0.3, size=n_items)

    if treatment == "bmad_ai":
        # 代理编排提升 -> 有效减少开发/评审耗时
        eff = cfg.orchestration_efficiency
        accept = cfg.ai_suggestion_accept_rate
        coverage = cfg.ai_assist_coverage
        gate = cfg.quality_gate_threshold

        # AI对覆盖部分的有效提速，受采纳率与门控阈值调制
        ai_factor = 1.0 + (eff - 1.0) * coverage * (0.5*accept + 0.5*gate)
        dev_time *= 1.0 / ai_factor
        review_time *= 1.0 / (ai_factor * 0.95)

        # 协作结构：更多跨角色评审 & 更高角色熵
        cross_ratio = 0.5 + 0.2*coverage*accept
        role_entropy = 1.0 + 0.25*coverage
        # 质量：失败率受门控与采纳率共同影响（过宽/过严均不利，做个U形近似）
        base_fail = 0.10
        delta = 0.06 * (abs(gate-0.75) + abs(accept-0.6))  # 偏离甜蜜点的惩罚
        change_failure_rate = max(0.03, min(0.18, base_fail + delta - 0.04*eff))

    else:
        # 传统职能分离，更多交接成本
        dev_time *= cfg.handoff_cost
        review_time *= 1.05 * cfg.handoff_cost
        cross_ratio = 0.35
        role_entropy = 0.9
        change_failure_rate = 0.12

    # WIP约束下的粗粒度排队近似
    q = 0.0
    lead_times = []
    wip_samples = []
    queue_share = []

    for t_dev, t_rev in zip(dev_time, review_time):
        wait = max(0.0, (q - cfg.wip_limit) * 0.2)
        total = wait + t_dev + (t_rev / max(1, cfg.review_capacity/4))
        lead_times.append(total)
        q = max(0.0, q + t_dev - 0.8)  # 简化的流出
        wip_samples.append(min(q, cfg.wip_limit + 2))
        queue_share.append(wait / total if total > 0 else 0.0)

    lead_time_median = float(np.median(lead_times))
    deploy_freq_per_week = float(7.0 / (np.mean(lead_times) + 1e-6))
    mttr_hours = float(np.clip(np.mean(lead_times) * 0.2, 0.5, 24.0))

    # 简化的silo_index：跨团队边比例的补数（这里用cross_ratio的函数近似）
    silo_index = float(max(0.0, 1.0 - (cross_ratio + 0.1)))

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": change_failure_rate,
        "metric/mttr_hours": mttr_hours,
        "metric/cross_discipline_review_ratio": float(cross_ratio),
        "metric/role_entropy": float(role_entropy),
        "metric/silo_index": silo_index,
        "metric/wip_avg": float(np.mean(wip_samples)),
        "metric/queue_time_share": float(np.mean(queue_share)),
        "meta/mode": "simulation",
        "meta/treatment": treatment
    }

def run_observational(dataset_path: str):
    # 需要包含以下列（示例）：
    # pr_open_ts, pr_merge_ts, is_failed_deploy, restore_hours, reviewer_role, author_role, files_roles
    df = pd.read_csv(dataset_path)
    lead_time = (pd.to_datetime(df["pr_merge_ts"]) - pd.to_datetime(df["pr_open_ts"])).dt.total_seconds() / 3600.0
    lead_time_median = float(lead_time.median())
    deploy_freq_per_week = float(df.shape[0] / max(1, (lead_time.sum()/24.0/7.0)))
    change_failure_rate = float(df["is_failed_deploy"].mean())
    mttr_hours = float(df["restore_hours"].replace(0, np.nan).median())

    # 跨角色评审
    cross = (df["reviewer_role"] != df["author_role"]).mean()

    # 角色熵（基于修改文件涉及的角色集合，分号分隔）
    def entropy(roles):
        parts = str(roles).split(";")
        vals, cnts = np.unique(parts, return_counts=True)
        p = cnts / cnts.sum()
        return float(-(p * np.log(p + 1e-9)).sum())
    role_entropy = float(df["files_roles"].apply(entropy).mean())

    # silo_index 需要评审网络；此处给出占位（若有网络数据，可用跨团队边比例计算）
    silo_index = float(max(0.0, 1.0 - (cross + 0.1)))

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": change_failure_rate,
        "metric/mttr_hours": mttr_hours,
        "metric/cross_discipline_review_ratio": float(cross),
        "metric/role_entropy": float(role_entropy),
        "metric/silo_index": float(silo_index),
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
    ap.add_argument("--wip_limit", type=int, default=3)
    ap.add_argument("--review_capacity", type=int, default=4)
    ap.add_argument("--handoff_cost", type=float, default=1.2)
    ap.add_argument("--ai_assist_coverage", type=float, default=0.6)
    ap.add_argument("--ai_suggestion_accept_rate", type=float, default=0.6)
    ap.add_argument("--quality_gate_threshold", type=float, default=0.7)
    ap.add_argument("--orchestration_efficiency", type=float, default=1.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode,
        dataset_path=args.dataset_path,
        seed=args.seed,
        wip_limit=args.wip_limit,
        review_capacity=args.review_capacity,
        handoff_cost=args.handoff_cost,
        ai_assist_coverage=args.ai_assist_coverage,
        ai_suggestion_accept_rate=args.ai_suggestion_accept_rate,
        quality_gate_threshold=args.quality_gate_threshold,
        orchestration_efficiency=args.orchestration_efficiency
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

    # 输出
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
