
# templates/bmad_devtest_quality/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass
rng = np.random.default_rng

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_path: str | None = None
    seed: int = 0
    treatments: tuple = ("bmad_integrated","silo_dev_qa")
    # 机制/参数
    quality_gate_threshold: float = 0.75
    ai_test_gen_coverage: float = 0.7
    test_coverage_target: float = 0.8
    mutation_score_target: float = 0.75
    flaky_rate: float = 0.1
    parallelism: int = 4
    pipeline_caching: float = 0.5
    change_size_loc: int = 400
    risk_profile: float = 0.5

def simulate_once(cfg: Config, treatment: str, r: np.random.Generator):
    n_items = 200
    # 变更工作量与风险
    size = cfg.change_size_loc
    risk = cfg.risk_profile
    base_dev = r.lognormal(mean=np.log(max(1, size/400)), sigma=0.4, size=n_items)
    base_test = r.lognormal(mean=0.0, sigma=0.35, size=n_items)

    # 质量/覆盖/变异/AI生成 对 test 有效强度
    quality_factor = (0.6*cfg.test_coverage_target + 0.4*cfg.mutation_score_target)
    ai_factor = (0.5*cfg.ai_test_gen_coverage + 0.5*quality_factor)

    if treatment == "bmad_integrated":
        # 一体化：AI生成+强门控+并行+缓存
        dev_time = base_dev * (1.0 - 0.05*cfg.pipeline_caching)
        test_time = base_test * (1.0 - 0.1*ai_factor)
        pipe_parallel_factor = max(1, cfg.parallelism)
        pipeline_duration_min = float((np.mean(dev_time + test_time) / pipe_parallel_factor) * (1.0 - 0.3*cfg.pipeline_caching) * 60)
        # 质量与失败率
        defect_leakage_ppk = float(max(0.1, (1.2 - 0.9*quality_factor - 0.6*ai_factor) * (0.6 + 0.4*risk)))
        cfr = float(np.clip(0.08 - 0.03*quality_factor + cfg.flaky_rate*0.2, 0.02, 0.2))
        test_pass_rate = float(0.9 - 0.3*cfg.flaky_rate + 0.05*ai_factor)
        test_cov = float(min(1.0, 0.5 + 0.6*cfg.test_coverage_target + 0.2*cfg.ai_test_gen_coverage - 0.1*risk))
        mut_score = float(min(1.0, 0.4 + 0.8*cfg.mutation_score_target + 0.05*cfg.ai_test_gen_coverage))
        flaky_fail_share = float(np.clip(cfg.flaky_rate * (1.0 - 0.3*cfg.pipeline_caching), 0.0, 1.0))
        cost_unit = float(1.0 + 0.5*cfg.parallelism/4 - 0.4*cfg.pipeline_caching + 0.2*cfg.ai_test_gen_coverage)
    else:
        # 传统分离：低门控+较低覆盖+较少并行与缓存
        dev_time = base_dev * 1.05
        test_time = base_test * 1.1
        pipeline_duration_min = float(np.mean(dev_time + test_time) * 60)
        defect_leakage_ppk = float(max(0.2, (1.5 - 0.6*quality_factor - 0.2*ai_factor) * (0.8 + 0.5*risk)))
        cfr = float(np.clip(0.12 + 0.3*cfg.flaky_rate, 0.05, 0.35))
        test_pass_rate = float(0.85 - 0.4*cfg.flaky_rate)
        test_cov = float(min(1.0, 0.45 + 0.4*cfg.test_coverage_target - 0.1*risk))
        mut_score = float(min(1.0, 0.35 + 0.5*cfg.mutation_score_target))
        flaky_fail_share = float(np.clip(cfg.flaky_rate * 1.1, 0.0, 1.0))
        cost_unit = float(1.0 + 0.3 + 0.1*cfg.parallelism/4)  # 并行少，但重跑多

    # DORA近似
    lead_times = dev_time + test_time + r.normal(0.0, 0.05, size=n_items)
    lead_time_median = float(np.median(lead_times))
    deploy_freq_per_week = float(7.0 / (np.mean(lead_times) + 1e-6))
    mttr_hours = float(np.clip(np.mean(lead_times) * (0.15 + 0.1*risk), 0.5, 24.0))

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": cfr,
        "metric/mttr_hours": mttr_hours,
        "metric/defect_leakage_ppk": defect_leakage_ppk,
        "metric/test_pass_rate": test_pass_rate,
        "metric/test_coverage": test_cov,
        "metric/mutation_score": mut_score,
        "metric/flaky_fail_share": flaky_fail_share,
        "metric/pipeline_duration_min": pipeline_duration_min,
        "metric/pipeline_cost_unit": cost_unit,
        "meta/mode": "simulation",
        "meta/treatment": treatment
    }

def run_observational(dataset_path: str):
    # 需要包含（示例）:
    # pr_open_ts, pr_merge_ts, is_failed_deploy, restore_hours,
    # test_pass_rate, test_coverage, mutation_score, flaky_fail_share,
    # pipeline_duration_min, pipeline_cost_unit, defect_leakage_ppk
    df = pd.read_csv(dataset_path)
    lead_time = (pd.to_datetime(df["pr_merge_ts"]) - pd.to_datetime(df["pr_open_ts"])).dt.total_seconds() / 3600.0
    lead_time_median = float(lead_time.median())
    deploy_freq_per_week = float(df.shape[0] / max(1, (lead_time.sum()/24.0/7.0)))
    change_failure_rate = float(df["is_failed_deploy"].mean())
    mttr_hours = float(df["restore_hours"].replace(0, np.nan).median())
    # 质量与管线
    pass_rate = float(df["test_pass_rate"].mean())
    coverage = float(df["test_coverage"].mean())
    mut = float(df["mutation_score"].mean())
    flaky = float(df["flaky_fail_share"].mean())
    pipe_min = float(df["pipeline_duration_min"].mean())
    cost = float(df["pipeline_cost_unit"].mean())
    leakage = float(df["defect_leakage_ppk"].mean())

    return {
        "metric/lead_time_median": lead_time_median,
        "metric/deploy_freq_per_week": deploy_freq_per_week,
        "metric/change_failure_rate": change_failure_rate,
        "metric/mttr_hours": mttr_hours,
        "metric/defect_leakage_ppk": leakage,
        "metric/test_pass_rate": pass_rate,
        "metric/test_coverage": coverage,
        "metric/mutation_score": mut,
        "metric/flaky_fail_share": flaky,
        "metric/pipeline_duration_min": pipe_min,
        "metric/pipeline_cost_unit": cost,
        "meta/mode": "observational",
        "meta/treatment": "unknown"
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["simulation","observational"], default="simulation")
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality_gate_threshold", type=float, default=0.75)
    ap.add_argument("--ai_test_gen_coverage", type=float, default=0.7)
    ap.add_argument("--test_coverage_target", type=float, default=0.8)
    ap.add_argument("--mutation_score_target", type=float, default=0.75)
    ap.add_argument("--flaky_rate", type=float, default=0.1)
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--pipeline_caching", type=float, default=0.5)
    ap.add_argument("--change_size_loc", type=int, default=400)
    ap.add_argument("--risk_profile", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode,
        dataset_path=args.dataset_path,
        seed=args.seed,
        quality_gate_threshold=args.quality_gate_threshold,
        ai_test_gen_coverage=args.ai_test_gen_coverage,
        test_coverage_target=args.test_coverage_target,
        mutation_score_target=args.mutation_score_target,
        flaky_rate=args.flaky_rate,
        parallelism=args.parallelism,
        pipeline_caching=args.pipeline_caching,
        change_size_loc=args.change_size_loc,
        risk_profile=args.risk_profile
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
