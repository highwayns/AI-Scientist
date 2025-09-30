
# templates/blockchain_audit_immutable/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass

rng = np.random.default_rng

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_path: str | None = None
    seed: int = 0
    treatments: tuple = ("chain_anchor","centralized_worm")
    anchor_frequency_sec: int = 60
    batch_size: int = 50
    chain_confirmations: int = 3
    tx_cost_unit: float = 1.0
    reorg_prob: float = 0.005
    adversary_strength: float = 0.4
    worm_immutability_factor: float = 0.85

def simulate_once(cfg: Config, treatment: str, r: np.random.Generator):
    # 生成一天的日志流（条/秒近似）
    seconds = 24*60*60
    base_rate = 5.0  # logs/sec
    logs = int(base_rate * seconds)
    # 对抗性篡改尝试
    tamper_trials = int(0.0005 * logs * (0.5 + cfg.adversary_strength))
    # 锚定成本与延迟
    if treatment == "chain_anchor":
        # 批量Merkle + 上链确认
        anchors = max(1, logs // max(1, cfg.batch_size))
        anchoring_cost = anchors * cfg.tx_cost_unit
        anchoring_latency = cfg.anchor_frequency_sec * (0.5 + 0.5* r.random()) + 30 * cfg.chain_confirmations
        # 检测率：锚定+确认数降低未检率；重组风险降低完整性
        base_detect = 0.95 - 0.1*np.tanh(cfg.batch_size/500)
        detect_rate = float(np.clip(base_detect * (1.0 - cfg.reorg_prob*cfg.chain_confirmations*0.5), 0.5, 0.99))
        fp_rate = float(np.clip(0.01 + 0.001*cfg.chain_confirmations, 0.0, 0.05))
        integrity = float(np.clip(0.7 + 0.25*detect_rate - 0.2*cfg.reorg_prob, 0.0, 1.0))
        legal_hold = float(np.clip(0.85 + 0.1*detect_rate - 0.05*cfg.reorg_prob, 0.0, 1.0))
    else:
        # 集中式WORM/签名链
        anchoring_cost = logs * 0.0
        anchoring_latency = float(5 + 10*r.random())
        # 检测率依赖不可改强度与管理员对抗
        detect_rate = float(np.clip(0.80*cfg.worm_immutability_factor - 0.2*cfg.adversary_strength, 0.2, 0.9))
        fp_rate = 0.008
        integrity = float(np.clip(0.6 + 0.3*cfg.worm_immutability_factor - 0.2*cfg.adversary_strength, 0.0, 1.0))
        legal_hold = float(np.clip(0.8 + 0.15*cfg.worm_immutability_factor, 0.0, 1.0))

    # MTTD ~ 与检测机制、批量与确认数相关
    mttd_hours = float(np.clip(anchoring_latency/3600.0 * (1.2 - 0.5*detect_rate), 0.01, 12.0))
    storage_overhead = float(np.clip(0.05 + (0.5 if treatment=="chain_anchor" else 0.02) + 0.0001*cfg.batch_size, 0.05, 1.0))
    throughput = float(base_rate * (1.0 - 0.05*(cfg.batch_size>200)))

    # 统计 proxy
    tp = int(detect_rate * tamper_trials)
    fn = tamper_trials - tp
    fp = int(fp_rate * logs * 1e-4)  # 极低误报基线
    # 单位成本：按天；可后续按实际币价换算
    cost_unit = float(anchoring_cost)

    return {
        "metric/detection_rate": float(detect_rate),
        "metric/false_positive_rate": float(fp_rate),
        "metric/audit_integrity_score": float(integrity),
        "metric/anchoring_latency_sec": float(anchoring_latency),
        "metric/anchoring_cost_unit": cost_unit,
        "metric/storage_overhead_ratio": storage_overhead,
        "metric/incident_mttd_hours": mttd_hours,
        "metric/legal_hold_success_rate": float(legal_hold),
        "metric/throughput_logs_per_sec": throughput,
        "meta/mode": "simulation",
        "meta/treatment": treatment
    }

def run_observational(dataset_path: str):
    # 期望列：timestamp, is_tamper_attempt, detected, is_false_positive, anchor_txid(optional), confirmations(optional)
    df = pd.read_csv(dataset_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    logs = len(df)
    tamper = int(df.get("is_tamper_attempt", pd.Series([0]*logs)).sum())
    detected = int(df.get("detected", pd.Series([0]*logs)).sum())
    false_pos = int(df.get("is_false_positive", pd.Series([0]*logs)).sum())

    detection_rate = float(detected / max(1, tamper))
    fp_rate = float(false_pos / max(1, logs))
    # 锚定延迟与成本（若无则占位为NaN）
    anchoring_latency = float(df.get("anchoring_latency_sec", pd.Series([np.nan]*logs)).mean())
    anchoring_cost = float(df.get("anchoring_cost_unit", pd.Series([np.nan]*logs)).mean())
    storage_overhead = float(df.get("storage_overhead_ratio", pd.Series([np.nan]*logs)).mean())
    mttd_hours = float(df.get("incident_mttd_hours", pd.Series([np.nan]*logs)).mean())
    legal_hold = float(df.get("legal_hold_success_rate", pd.Series([np.nan]*logs)).mean())
    throughput = float(df.get("throughput_logs_per_sec", pd.Series([np.nan]*logs)).mean())
    integrity = float(df.get("audit_integrity_score", pd.Series([np.nan]*logs)).mean())

    return {
        "metric/detection_rate": detection_rate,
        "metric/false_positive_rate": fp_rate,
        "metric/audit_integrity_score": integrity,
        "metric/anchoring_latency_sec": anchoring_latency,
        "metric/anchoring_cost_unit": anchoring_cost,
        "metric/storage_overhead_ratio": storage_overhead,
        "metric/incident_mttd_hours": mttd_hours,
        "metric/legal_hold_success_rate": legal_hold,
        "metric/throughput_logs_per_sec": throughput,
        "meta/mode": "observational",
        "meta/treatment": "unknown"
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["simulation","observational"], default="simulation")
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor_frequency_sec", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--chain_confirmations", type=int, default=3)
    ap.add_argument("--tx_cost_unit", type=float, default=1.0)
    ap.add_argument("--reorg_prob", type=float, default=0.005)
    ap.add_argument("--adversary_strength", type=float, default=0.4)
    ap.add_argument("--worm_immutability_factor", type=float, default=0.85)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode,
        dataset_path=args.dataset_path,
        seed=args.seed,
        anchor_frequency_sec=args.anchor_frequency_sec,
        batch_size=args.batch_size,
        chain_confirmations=args.chain_confirmations,
        tx_cost_unit=args.tx_cost_unit,
        reorg_prob=args.reorg_prob,
        adversary_strength=args.adversary_strength,
        worm_immutability_factor=args.worm_immutability_factor
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

    series = np.array([[row[k] for k in [
        "metric/detection_rate","metric/false_positive_rate",
        "metric/anchoring_latency_sec","metric/anchoring_cost_unit"
    ]] for row in rows], dtype=float)
    np.save(os.path.join(args.out_dir, "series.npy"), series)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "per_run_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({k: rows[0][k] for k in rows[0].keys()}, f, indent=2)
