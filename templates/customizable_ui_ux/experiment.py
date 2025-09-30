
# templates/customizable_ui_ux/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass
rng = np.random.default_rng

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_path: str | None = None
    seed: int = 0
    treatments: tuple = ("adaptive_ui","static_ui")
    personalization_depth: float = 0.7
    latency_budget_ms: int = 150
    accessibility_contrast: float = 0.7
    font_scale: float = 1.2
    layout_density: float = 1.0
    device_class: str = "mobile"
    input_mix_touch_ratio: float = 0.7
    context_switch_rate: float = 0.4

def simulate_once(cfg: Config, treatment: str, r: np.random.Generator):
    # 合成用户×任务×场景
    n_users = 120
    # 用户多样性：视力/运动/经验（0~1）
    vision = r.beta(2,3, size=n_users)   # 趋向较弱视力比例
    motor  = r.beta(3,2, size=n_users)
    exp    = r.beta(2.5,2.5, size=n_users)

    # 任务复杂度与场景负荷
    task_complex = r.uniform(0.3, 0.9, size=n_users)
    env_glare = r.uniform(0.0, 1.0, size=n_users) * cfg.context_switch_rate  # 强光/移动带来的可视负担

    # UI策略影响项
    if treatment == "adaptive_ui":
        # 自适应提升：随个性化深度与可达性增强
        adapt_gain = 0.15 + 0.35*cfg.personalization_depth + 0.2*cfg.accessibility_contrast
        latency_penalty = max(0, (cfg.latency_budget_ms - 120)/1000.0)  # 超预算带来交互延迟
        contrast_effect = cfg.accessibility_contrast - 0.5
        font_effect = (cfg.font_scale - 1.0)*0.2
        layout_effect = 0.1*(1.1 - abs(cfg.layout_density-1.0))  # 极端密度不利
    else:
        adapt_gain = 0.05
        latency_penalty = 0.02
        contrast_effect = -0.05
        font_effect = 0.0
        layout_effect = 0.0

    # 指标合成（简化近似）
    # 任务成功率：受经验、适配增益、对比度与场景影响
    success = 0.6 + 0.25*exp + 0.3*adapt_gain + 0.2*contrast_effect - 0.25*env_glare - 0.2*task_complex
    success = np.clip(success, 0.05, 0.99)

    # 完成时长（秒）：受任务复杂度、经验、适配增益、延迟惩罚、字体/布局影响
    base_time = 40 + 60*task_complex - 20*exp - 10*adapt_gain + 8*latency_penalty - 4*layout_effect - 5*font_effect
    base_time *= (1.0 + (0.15 if cfg.device_class=='mobile' else 0.05))
    completion_time = np.clip(base_time, 5, 300)

    # 错误率：受视觉、运动、触控比例、对比度与字号
    error = 0.25 + 0.25*(1-vision) + 0.15*(1-motor) + 0.1*cfg.input_mix_touch_ratio - 0.2*contrast_effect - 0.05*font_effect - 0.1*adapt_gain
    error += 0.1*env_glare
    error = np.clip(error, 0.01, 0.8)

    # SUS（0~100）与 NASA-TLX（0~100, 越低越好）
    sus = 55 + 25*adapt_gain + 10*exp - 15*latency_penalty - 10*error + 5*layout_effect + 3*font_effect
    sus = np.clip(sus, 20, 95)
    tlx = 60 - 20*adapt_gain - 10*contrast_effect - 5*font_effect + 10*env_glare + 10*task_complex + 5*latency_penalty
    tlx = np.clip(tlx, 5, 95)

    # 交互与视口指标
    interaction_latency = 120 + 3*cfg.latency_budget_ms - 100*adapt_gain
    interaction_latency = float(np.clip(interaction_latency, 20, 800))
    scroll_depth_ratio = float(np.clip(0.5 + 0.3*task_complex - 0.2*adapt_gain - 0.1*font_effect, 0.05, 1.2))
    viewport_util = float(np.clip(0.6 + 0.2*layout_effect - 0.1*font_effect, 0.2, 1.0))
    a11y_events = float(np.clip(2.0 + 3.0*(1-vision.mean()) + 1.0*(env_glare.mean()) + (0.5 - contrast_effect), 0.1, 10.0))

    return {
        "metric/task_success_rate": float(success.mean()),
        "metric/avg_completion_time_s": float(np.mean(completion_time)),
        "metric/error_rate": float(error.mean()),
        "metric/sus_score": float(np.mean(sus)),
        "metric/nasa_tlx": float(np.mean(tlx)),
        "metric/scroll_depth_ratio": float(scroll_depth_ratio),
        "metric/viewport_utilization": float(viewport_util),
        "metric/interaction_latency_ms": float(interaction_latency),
        "metric/accessibility_events_per_min": float(a11y_events),
        "meta/mode": "simulation",
        "meta/treatment": treatment
    }

def run_observational(dataset_path: str):
    # 期望列：success, completion_time_s, error, sus, tlx, scroll_depth_ratio, viewport_utilization, interaction_latency_ms, a11y_events
    df = pd.read_csv(dataset_path)
    return {
        "metric/task_success_rate": float(df["success"].mean()),
        "metric/avg_completion_time_s": float(df["completion_time_s"].mean()),
        "metric/error_rate": float(df["error"].mean()),
        "metric/sus_score": float(df["sus"].mean()),
        "metric/nasa_tlx": float(df["tlx"].mean()),
        "metric/scroll_depth_ratio": float(df.get("scroll_depth_ratio", np.nan).mean()),
        "metric/viewport_utilization": float(df.get("viewport_utilization", np.nan).mean()),
        "metric/interaction_latency_ms": float(df.get("interaction_latency_ms", np.nan).mean()),
        "metric/accessibility_events_per_min": float(df.get("a11y_events", np.nan).mean()),
        "meta/mode": "observational",
        "meta/treatment": "unknown"
    }

if __name__ == "__main__":
    import argparse, os, json, numpy as np, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["simulation","observational"], default="simulation")
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--personalization_depth", type=float, default=0.7)
    ap.add_argument("--latency_budget_ms", type=int, default=150)
    ap.add_argument("--accessibility_contrast", type=float, default=0.7)
    ap.add_argument("--font_scale", type=float, default=1.2)
    ap.add_argument("--layout_density", type=float, default=1.0)
    ap.add_argument("--device_class", choices=["mobile","tablet","desktop"], default="mobile")
    ap.add_argument("--input_mix_touch_ratio", type=float, default=0.7)
    ap.add_argument("--context_switch_rate", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode,
        dataset_path=args.dataset_path,
        seed=args.seed,
        personalization_depth=args.personalization_depth,
        latency_budget_ms=args.latency_budget_ms,
        accessibility_contrast=args.accessibility_contrast,
        font_scale=args.font_scale,
        layout_density=args.layout_density,
        device_class=args.device_class,
        input_mix_touch_ratio=args.input_mix_touch_ratio,
        context_switch_rate=args.context_switch_rate
    )

    r = rng(cfg.seed)
    rows = []
    if cfg.mode == "observational" and cfg.dataset_path:
        rows.append(run_observational(cfg.dataset_path))
    else:
        for treatment in cfg.treatments:
            rows.append(simulate_once(cfg, treatment, r))

    series = np.array([[row[k] for k in [
        "metric/task_success_rate","metric/avg_completion_time_s",
        "metric/error_rate","metric/sus_score"
    ]] for row in rows], dtype=float)
    np.save(os.path.join(args.out_dir, "series.npy"), series)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "per_run_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({k: rows[0][k] for k in rows[0].keys()}, f, indent=2)
