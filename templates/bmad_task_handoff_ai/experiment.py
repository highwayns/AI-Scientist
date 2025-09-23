
# templates/bmad_task_handoff_ai/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass
rng = np.random.default_rng

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_path: str | None = None
    seed: int = 0
    treatments: tuple = ("bmad_ai_handoff", "silo_manual_handoff")
    # 切分/交接/AI能力
    task_granularity: float = 1.0
    handoff_package_completeness: float = 0.8
    rag_quality: float = 0.8
    llm_window_tokens: int = 32000
    handoff_chain_length: int = 3
    quality_gate_threshold: float = 0.7
    # 流程约束
    wip_limit: int = 3
    review_capacity: int = 4

def simulate_once(cfg: Config, treatment: str, r: np.random.Generator):
    n_items = 240
    # 基础开发/评审耗时（对数正态）
    dev_time = r.lognormal(mean=0.0, sigma=0.4, size=n_items)
    review_time = r.lognormal(mean=-0.1, sigma=0.3, size=n_items)

    # 交接损失与缺陷模型（简化）：损失率~ 任务粒度↑、链长↑、完整度↓、RAG↓、窗口↓
    window_factor = min(1.0, cfg.llm_window_tokens/32000.0)  # 32k作为基准
    base_loss = 0.20*cfg.task_granularity + 0.03*cfg.handoff_chain_length
    base_loss *= (1.0 - 0.6*cfg.handoff_package_completeness) * (1.0 - 0.5*cfg.rag_quality) * (1.0 - 0.4*window_factor)
    base_loss = np.clip(base_loss, 0.01, 0.5)

    if treatment == "bmad_ai_handoff":
        # 结构化上下文包 + AI检索压缩 + 门控
        eff = 1.0 + 0.25*cfg.rag_quality*window_factor + 0.15*cfg.handoff_package_completeness
        dev_time *= 1.0/eff
        review_time *= 1.0/(eff*0.95)
        handoff_loss = base_loss * 0.5
        defects_per_handoff = 0.10 * handoff_loss * (1.0 - cfg.quality_gate_threshold*0.6)
        cross_ratio = 0.5 + 0.1*cfg.handoff_package_completeness
        role_entropy = 1.0 + 0.2*cfg.rag_quality
        token_cost_per_task = 0.5 + 0.5*window_factor  # 简化：大窗口更贵
    else:
        dev_time *= 1.15
        review_time *= 1.10
        handoff_loss = base_loss * 1.2 + 0.05
        defects_per_handoff = 0.12 * handoff_loss
        cross_ratio = 0.35
        role_entropy = 0.9
        token_cost_per_task = 0.4  # 无大模型上下文成本

    # WIP约束下的粗粒度排队
    q = 0.0
    lead_times, wip_samples, queue_share = [], [], []
    for t_dev, t_rev in zip(dev_time, review_time):
        wait = max(0.0, (q - cfg.wip_limit) * 0.2)
        total = wait + t_dev + (t_rev / max(1, cfg.review_capacity/4))
        lead_times.append(total)
        q = max(0.0, q + t_dev - 0.8)
        wip_samples.append(min(q, cfg.wip_limit + 2))
        queue_share.append(wait / total if total > 0 else 0.0)

    lead_time_median = float(np.median(lead_times))
    deploy_freq_per_week = float(7.0 / (np.mean(lead_times) + 1e-6))
    change_failure_rate = float(np.clip(0.06 + 0.8*defects_per_handoff, 0.02, 0.25))
    mttr_hours = float(np.clip(np.mean(lead_times) * 0.2, 0.5, 24.0))

    # silo_index：用跨角色比例的补数近似
    silo_index = float(max(0.0, 1.0 - (cross_ratio + 0.1)))

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": change_failure_rate,
        "metric/mttr_hours": mttr_hours,
        "metric/handoff_loss_rate": float(handoff_loss),
        "metric/defects_per_handoff": float(defects_per_handoff),
        "metric/context_reuse_ratio": float(1.0 - handoff_loss),
        "metric/token_cost_per_task": float(token_cost_per_task),
        "metric/cross_role_review_ratio": float(cross_ratio),
        "metric/role_entropy": float(role_entropy),
        "metric/silo_index": float(silo_index),
        "meta/mode": "simulation",
        "meta/treatment": treatment
    }

def run_observational(dataset_path: str):
    # 需要包含以下列（示例）：
    # pr_open_ts, pr_merge_ts, is_failed_deploy, restore_hours,
    # reviewer_role, author_role, files_roles, handoff_chain_length,
    # context_missing_fields, context_total_fields, tokens_used
    df = pd.read_csv(dataset_path)
    lead_time = (pd.to_datetime(df["pr_merge_ts"]) - pd.to_datetime(df["pr_open_ts"])).dt.total_seconds() / 3600.0
    lead_time_median = float(lead_time.median())
    deploy_freq_per_week = float(df.shape[0] / max(1, (lead_time.sum()/24.0/7.0)))
    change_failure_rate = float(df["is_failed_deploy"].mean())
    mttr_hours = float(df["restore_hours"].replace(0, np.nan).median())

    # 交接损失率与缺陷
    handoff_loss = float((df["context_missing_fields"] / df["context_total_fields"]).clip(0,1).mean())
    defects_per_handoff = float((df.get("defects_count", pd.Series([0]*len(df))) / 
                                 df.get("handoff_chain_length", pd.Series([1]*len(df))).replace(0,1)).mean())
    context_reuse_ratio = float(1.0 - handoff_loss)
    token_cost_per_task = float(df.get("tokens_used", pd.Series([0]*len(df))).mean())

    # 跨角色评审
    cross = (df["reviewer_role"] != df["author_role"]).mean()

    # 角色熵（基于修改文件涉及角色集合，分号分隔）
    def entropy(roles):
        parts = str(roles).split(";")
        vals, cnts = np.unique(parts, return_counts=True)
        p = cnts / cnts.sum()
        return float(-(p * np.log(p + 1e-9)).sum())
    role_entropy = float(df["files_roles"].apply(entropy).mean())

    silo_index = float(max(0.0, 1.0 - (cross + 0.1)))

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": change_failure_rate,
        "metric/mttr_hours": mttr_hours,
        "metric/handoff_loss_rate": handoff_loss,
        "metric/defects_per_handoff": defects_per_handoff,
        "metric/context_reuse_ratio": context_reuse_ratio,
        "metric/token_cost_per_task": token_cost_per_task,
        "metric/cross_role_review_ratio": float(cross),
        "metric/role_entropy": float(role_entropy),
        "metric/silo_index": float(silo_index),
        "meta/mode": "observational",
        "meta/treatment": "unknown"
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["simulation","observational"], default="simulation")
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--treatment_bmad", default="bmad_ai_handoff")
    ap.add_argument("--treatment_silo", default="silo_manual_handoff")
    ap.add_argument("--task_granularity", type=float, default=1.0)
    ap.add_argument("--handoff_package_completeness", type=float, default=0.8)
    ap.add_argument("--rag_quality", type=float, default=0.8)
    ap.add_argument("--llm_window_tokens", type=int, default=32000)
    ap.add_argument("--handoff_chain_length", type=int, default=3)
    ap.add_argument("--quality_gate_threshold", type=float, default=0.7)
    ap.add_argument("--wip_limit", type=int, default=3)
    ap.add_argument("--review_capacity", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode,
        dataset_path=args.dataset_path,
        seed=args.seed,
        task_granularity=args.task_granularity,
        handoff_package_completeness=args.handoff_package_completeness,
        rag_quality=args.rag_quality,
        llm_window_tokens=args.llm_window_tokens,
        handoff_chain_length=args.handoff_chain_length,
        quality_gate_threshold=args.quality_gate_threshold,
        wip_limit=args.wip_limit,
        review_capacity=args.review_capacity
    )

    r = rng(cfg.seed)
    rows = []

    if cfg.mode == "observational" and cfg.dataset_path:
        res = run_observational(cfg.dataset_path)
        rows.append(res)
    else:
        for treatment in (args.treatment_bmad, args.treatment_silo):
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
