
# templates/edge_realtime_pipeline/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass
rng = np.random.default_rng

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_path: str | None = None
    seed: int = 0
    treatments: tuple = ("optimized_pipeline","baseline_pipeline")
    batch_size: int = 4
    window_ms: int = 10
    fusion_level: int = 1
    quant_bits: int = 8
    zero_copy: bool = True
    numa_affinity: bool = True
    rt_priority: bool = True
    cache_policy: str = "lru"
    warm_pool_size: int = 4
    cold_start_ms: int = 120
    network_rtt_ms: int = 20
    device_class: str = "edge_cpu"
    stream_count: int = 8
    load_factor: float = 1.0

def synth_once(cfg: Config, treatment: str, r: np.random.Generator):
    # 设备基线吞吐/延迟
    if cfg.device_class == "edge_cpu":
        base_service_ms = 12.0
        base_energy_mj = 3.0
    elif cfg.device_class == "edge_gpu":
        base_service_ms = 6.0
        base_energy_mj = 2.5
    else:
        base_service_ms = 4.0
        base_energy_mj = 1.6

    # 优化与配置影响
    fusion_gain = 1.0 + 0.12*cfg.fusion_level if treatment=="optimized_pipeline" else 1.0
    zero_copy_gain = 1.10 if (treatment=="optimized_pipeline" and cfg.zero_copy) else 1.0
    quant_gain = 1.0 + (0.25 if (treatment=="optimized_pipeline" and cfg.quant_bits<=8) else (0.12 if cfg.quant_bits==16 else 0.0))
    sched_gain = 1.08 if (treatment=="optimized_pipeline" and cfg.rt_priority) else 1.0
    numa_gain = 1.05 if (treatment=="optimized_pipeline" and cfg.numa_affinity) else 1.0
    batch_penalty = max(0.0, (cfg.batch_size-1)*0.02)  # 批太大带来尾延迟上涨
    window_penalty = max(0.0, (cfg.window_ms-10)*0.01)

    service_ms = base_service_ms / (fusion_gain*zero_copy_gain*quant_gain*sched_gain*numa_gain)
    service_ms *= (1.0 + batch_penalty + window_penalty)

    # 排队：M/M/k 近似（简化）
    k = max(1, min(cfg.stream_count, cfg.warm_pool_size if treatment=="optimized_pipeline" else max(1, cfg.warm_pool_size-2)))
    arrival_rate = cfg.load_factor * cfg.stream_count / (service_ms/1000.0 + cfg.network_rtt_ms/1000.0/cfg.batch_size)
    mu = 1000.0/service_ms
    rho = min(0.98, arrival_rate/(k*mu))
    # 等待时间近似（Erlang C 简化为 rho^((2k)) 级别项）
    wait_ms = (service_ms * (rho**(2*k))) / max(0.05, (1.0 - rho))

    # 冷启动：基线更频繁
    cold_rate = max(0.0, (0.03 + 0.02*cfg.load_factor) * (0.5 if treatment=="optimized_pipeline" else 1.5) * (1.0/(1+cfg.warm_pool_size)))
    cold_ms = cold_rate * cfg.cold_start_ms

    # 网络抖动
    rtt_jitter = r.normal(0.0, 0.15*cfg.network_rtt_ms)
    path_ms = max(0.0, cfg.network_rtt_ms + rtt_jitter)

    # 延迟分布（p50/p95/p99）
    p50 = service_ms + wait_ms*0.3 + path_ms
    p95 = p50 * (1.25 + 0.1*batch_penalty + 0.05*window_penalty) + cold_ms
    p99 = p50 * (1.45 + 0.15*batch_penalty + 0.08*window_penalty) + 1.5*cold_ms
    jitter = max(0.0, p95 - p50)

    # 吞吐与丢弃
    throughput = (k*mu) * (1.0 - min(0.2, rho*0.1)) / (1.0 + batch_penalty)
    drop = max(0.0, rho - 0.95) * 2.0

    # 资源与能耗
    cpu_util = min(0.99, 0.4 + 0.6*rho + (0.05 if treatment=="baseline_pipeline" else -0.03))
    mem_bw = 3.0 * (1.0 + 0.3*cfg.stream_count/8.0) * (1.0 - (0.1 if (treatment=='optimized_pipeline' and cfg.zero_copy) else 0.0))
    energy = base_energy_mj * (service_ms/base_service_ms) * (1.0 - (0.15 if (treatment=='optimized_pipeline' and cfg.quant_bits<=8) else 0.0))

    # 缓存命中（模型/特征）：优化+策略更好
    if treatment=="optimized_pipeline":
        hit = 0.75 if cfg.cache_policy=="lru" else (0.8 if cfg.cache_policy=="2q" else 0.6)
    else:
        hit = 0.55 if cfg.cache_policy=="lru" else 0.5

    return {
        "metric/latency_p50_ms": float(max(0.1, p50)),
        "metric/latency_p95_ms": float(max(0.1, p95)),
        "metric/latency_p99_ms": float(max(0.1, p99)),
        "metric/jitter_ms": float(max(0.0, jitter)),
        "metric/throughput_fps": float(max(0.1, throughput)),
        "metric/drop_rate": float(max(0.0, min(1.0, drop))),
        "metric/cpu_util": float(cpu_util),
        "metric/memory_bw_gbps": float(mem_bw),
        "metric/energy_per_inf_mj": float(max(0.05, energy)),
        "metric/cache_hit_ratio": float(hit),
        "metric/cold_start_rate": float(min(1.0, cold_rate)),
        "meta/mode": "simulation",
        "meta/treatment": treatment
    }

def run_observational(dataset_path: str):
    # 需要列：lat_p50, lat_p95, lat_p99, jitter, throughput, drop, cpu_util, mem_bw_gbps, energy_mj, cache_hit, cold_rate
    df = pd.read_csv(dataset_path)
    return {
        "metric/latency_p50_ms": float(df["lat_p50"].mean()),
        "metric/latency_p95_ms": float(df["lat_p95"].mean()),
        "metric/latency_p99_ms": float(df["lat_p99"].mean()),
        "metric/jitter_ms": float(df["jitter"].mean()),
        "metric/throughput_fps": float(df["throughput"].mean()),
        "metric/drop_rate": float(df["drop"].mean()),
        "metric/cpu_util": float(df["cpu_util"].mean()),
        "metric/memory_bw_gbps": float(df["mem_bw_gbps"].mean()),
        "metric/energy_per_inf_mj": float(df["energy_mj"].mean()),
        "metric/cache_hit_ratio": float(df["cache_hit"].mean()),
        "metric/cold_start_rate": float(df["cold_rate"].mean()),
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
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--window_ms", type=int, default=10)
    ap.add_argument("--fusion_level", type=int, default=1)
    ap.add_argument("--quant_bits", type=int, default=8)
    ap.add_argument("--zero_copy", action="store_true")
    ap.add_argument("--numa_affinity", action="store_true")
    ap.add_argument("--rt_priority", action="store_true")
    ap.add_argument("--cache_policy", choices=["none","lru","2q"], default="lru")
    ap.add_argument("--warm_pool_size", type=int, default=4)
    ap.add_argument("--cold_start_ms", type=int, default=120)
    ap.add_argument("--network_rtt_ms", type=int, default=20)
    ap.add_argument("--device_class", choices=["edge_cpu","edge_gpu","edge_npu"], default="edge_cpu")
    ap.add_argument("--stream_count", type=int, default=8)
    ap.add_argument("--load_factor", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode, dataset_path=args.dataset_path, seed=args.seed,
        batch_size=args.batch_size, window_ms=args.window_ms, fusion_level=args.fusion_level,
        quant_bits=args.quant_bits, zero_copy=args.zero_copy, numa_affinity=args.numa_affinity,
        rt_priority=args.rt_priority, cache_policy=args.cache_policy, warm_pool_size=args.warm_pool_size,
        cold_start_ms=args.cold_start_ms, network_rtt_ms=args.network_rtt_ms, device_class=args.device_class,
        stream_count=args.stream_count, load_factor=args.load_factor
    )

    r = rng(cfg.seed)
    rows = []
    if cfg.mode == "observational" and cfg.dataset_path:
        rows.append(run_observational(cfg.dataset_path))
    else:
        for t in cfg.treatments:
            rows.append(synth_once(cfg, t, r))

    series = np.array([[row[k] for k in [
        "metric/latency_p50_ms","metric/latency_p95_ms","metric/latency_p99_ms"
    ]] for row in rows], dtype=float)
    np.save(os.path.join(args.out_dir, "series.npy"), series)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "per_run_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({k: rows[0][k] for k in rows[0].keys()}, f, indent=2)
