
# templates/contactless_vitals_robustness/experiment.py
import argparse, os, json, numpy as np, pandas as pd
from dataclasses import dataclass
rng = np.random.default_rng

def bland_altman_limits(ref, pred):
    diff = pred - ref
    m = diff.mean()
    s = diff.std()
    return float(m - 1.96*s), float(m + 1.96*s)

@dataclass
class Config:
    mode: str = "simulation"  # or 'observational'
    dataset_path: str | None = None
    seed: int = 0
    treatments: tuple = ("robust_pipeline","baseline_pipeline")
    modality: str = "rgb"
    distance_m: float = 1.0
    illum_lux: float = 100.0
    motion_level: float = 0.2
    skin_tone: float = 0.5
    occlusion_ratio: float = 0.1
    wqm_threshold: float = 0.6
    roi_adapt_strength: float = 0.8
    temporal_smooth_strength: float = 0.4
    spectral_peak_robust: bool = True
    channels: int = 3
    fps: int = 30

def synth_once(cfg: Config, treatment: str, r: np.random.Generator):
    n = 600  # 20s@30fps
    # 真实HR/RR（bpm）
    true_hr = r.normal(72, 6)
    true_rr = r.normal(14, 2)
    t = np.arange(n)/cfg.fps

    # 生成理想波形并注入噪声（运动/光照/肤色/遮挡/距离）
    base_hr_sig = np.sin(2*np.pi*(true_hr/60.0)*t)
    base_rr_sig = np.sin(2*np.pi*(true_rr/60.0)*t + 0.3)

    # SNR基线（不同模态）
    if cfg.modality == "rgb":
        snr0 = 8.0 - 3.0*cfg.skin_tone - 1.5*np.log1p(cfg.distance_m)
        power_mw = 500
    elif cfg.modality == "nir":
        snr0 = 10.0 - 1.5*np.log1p(cfg.distance_m)
        power_mw = 650
    elif cfg.modality == "thermal":
        snr0 = 9.0 - 2.0*np.log1p(cfg.distance_m)
        power_mw = 800
    elif cfg.modality == "mmwave":
        snr0 = 11.0 - 1.0*np.log1p(cfg.distance_m)
        power_mw = 1200
    else:  # fusion（简化为加性获益与功耗叠加折扣）
        snr0 = 12.0 - 1.0*np.log1p(cfg.distance_m) - 1.0*cfg.skin_tone
        power_mw = 1000

    # 环境与运动退化
    illum_pen = 2.0*np.exp(-cfg.illum_lux/300.0)  # 低照更差
    motion_pen = 6.0*cfg.motion_level
    occ_pen = 5.0*cfg.occlusion_ratio
    snr_db = snr0 - illum_pen - motion_pen - occ_pen
    snr_db = float(max(-5.0, snr_db))

    noise = r.normal(0, 10**(-snr_db/20.0), size=n)
    signal = 0.6*base_hr_sig + 0.4*base_rr_sig + noise

    # 管线差异
    if treatment == "robust_pipeline":
        # WQM门控：劣质段替换为插值/抑制
        wqm = 1.0 - (0.4*cfg.motion_level + 0.3*occ_pen/5.0 + 0.3*illum_pen/2.0)
        wqm = np.clip(wqm, 0.0, 1.0)
        mask = wqm < cfg.wqm_threshold
        if mask:
            # 简化处理：全局增益降低噪声
            signal = signal * (1.0 + 0.2*cfg.roi_adapt_strength) + r.normal(0, 0.5*noise.std(), size=n) * (0.5 - 0.5*cfg.roi_adapt_strength)
        # 时域平滑（避免过度）
        alpha = np.clip(0.2 + 0.6*cfg.temporal_smooth_strength, 0.2, 0.8)
        # 简单双边滑动平均
        k = int(3 + 12*cfg.temporal_smooth_strength)
        if k%2==0: k+=1
        kernel = np.ones(k)/k
        signal = np.convolve(signal, kernel, mode='same')
        # 频域稳健峰：降低假峰
        peak_bias = -0.5 if cfg.spectral_peak_robust else 0.0
    else:
        alpha = 0.2
        peak_bias = 0.0

    # 估计HR/RR（用频域峰近似）
    # 计算功率谱
    freq = np.fft.rfftfreq(n, d=1.0/cfg.fps)
    psd = np.abs(np.fft.rfft(signal))**2
    # HR频带 [0.7, 3] Hz；RR频带 [0.1, 0.6] Hz
    def band_peak(blo, bhi):
        mask = (freq>=blo) & (freq<=bhi)
        idx = np.argmax(psd[mask])
        f = freq[mask][idx]
        return f
    f_hr = band_peak(0.7, 3.0)
    f_rr = band_peak(0.1, 0.6)
    est_hr = 60.0*(f_hr + peak_bias*0.01)
    est_rr = 60.0*(f_rr + peak_bias*0.005)

    # 误差与覆盖/失效（SNR过低视为失效）
    hr_rmse = float(np.sqrt((est_hr-true_hr)**2))
    rr_rmse = float(np.sqrt((est_rr-true_rr)**2))
    hr_mae = float(abs(est_hr-true_hr))
    failure = float(1.0 if snr_db < 0.5 else 0.0)
    coverage = float(1.0 - failure)

    # HRV偏差（平滑过度会放大偏差）
    hrv_bias = float(0.15*cfg.temporal_smooth_strength - 0.05*(snr_db/10.0))

    # 延迟与功耗近似
    latency_ms = float(40 + 100*(1.0/cfg.fps) + (8 if treatment=='robust_pipeline' else 4))
    power_mw = float(power_mw * (1.1 if treatment=='robust_pipeline' else 1.0))

    loA_low, loA_high = bland_altman_limits(np.array([true_hr]), np.array([est_hr]))

    return {
        "metric/hr_rmse_bpm": hr_rmse,
        "metric/rr_rmse_bpm": rr_rmse,
        "metric/hr_mae_bpm": hr_mae,
        "metric/hrv_sdnn_bias": hrv_bias,
        "metric/coverage_rate": coverage,
        "metric/failure_rate": failure,
        "metric/snr_db": snr_db,
        "metric/latency_ms": latency_ms,
        "metric/power_mw": power_mw,
        "metric/bland_altman_loA_hr_low": loA_low,
        "metric/bland_altman_loA_hr_high": loA_high,
        "meta/mode": "simulation",
        "meta/treatment": treatment
    }

def run_observational(dataset_path: str):
    # 期望列：ref_hr, ref_rr, pred_hr, pred_rr（可多行样本）；可选：snr_db, latency_ms, power_mw, modality, skin_tone, distance_m, illum_lux, motion_level, occlusion_ratio
    df = pd.read_csv(dataset_path)
    ref_hr = df["ref_hr"].values
    pred_hr = df["pred_hr"].values
    ref_rr = df["ref_rr"].values
    pred_rr = df["pred_rr"].values

    hr_rmse = float(np.sqrt(np.mean((pred_hr-ref_hr)**2)))
    rr_rmse = float(np.sqrt(np.mean((pred_rr-ref_rr)**2)))
    hr_mae = float(np.mean(np.abs(pred_hr-ref_hr)))
    coverage = float((~np.isnan(pred_hr)).mean())
    failure = float(1.0 - coverage)
    snr = float(df.get("snr_db", pd.Series([np.nan]*len(df))).mean())
    latency = float(df.get("latency_ms", pd.Series([np.nan]*len(df))).mean())
    power = float(df.get("power_mw", pd.Series([np.nan]*len(df))).mean())

    # HRV偏差需有sdnn列，否则置NaN
    hrv_bias = float(df.get("sdnn_bias", pd.Series([np.nan]*len(df))).mean())

    from math import isnan
    loA_low, loA_high = (np.nan, np.nan)
    if len(ref_hr) > 1:
        diff = pred_hr - ref_hr
        m = diff.mean(); s = diff.std()
        loA_low, loA_high = float(m - 1.96*s), float(m + 1.96*s)

    return {
        "metric/hr_rmse_bpm": hr_rmse,
        "metric/rr_rmse_bpm": rr_rmse,
        "metric/hr_mae_bpm": hr_mae,
        "metric/hrv_sdnn_bias": hrv_bias,
        "metric/coverage_rate": coverage,
        "metric/failure_rate": failure,
        "metric/snr_db": snr,
        "metric/latency_ms": latency,
        "metric/power_mw": power,
        "metric/bland_altman_loA_hr_low": loA_low,
        "metric/bland_altman_loA_hr_high": loA_high,
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
    ap.add_argument("--modality", choices=["rgb","nir","thermal","mmwave","fusion"], default="rgb")
    ap.add_argument("--distance_m", type=float, default=1.0)
    ap.add_argument("--illum_lux", type=float, default=100.0)
    ap.add_argument("--motion_level", type=float, default=0.2)
    ap.add_argument("--skin_tone", type=float, default=0.5)
    ap.add_argument("--occlusion_ratio", type=float, default=0.1)
    ap.add_argument("--wqm_threshold", type=float, default=0.6)
    ap.add_argument("--roi_adapt_strength", type=float, default=0.8)
    ap.add_argument("--temporal_smooth_strength", type=float, default=0.4)
    ap.add_argument("--spectral_peak_robust", action="store_true")
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(
        mode=args.mode, dataset_path=args.dataset_path, seed=args.seed,
        modality=args.modality, distance_m=args.distance_m, illum_lux=args.illum_lux,
        motion_level=args.motion_level, skin_tone=args.skin_tone, occlusion_ratio=args.occlusion_ratio,
        wqm_threshold=args.wqm_threshold, roi_adapt_strength=args.roi_adapt_strength,
        temporal_smooth_strength=args.temporal_smooth_strength, spectral_peak_robust=args.spectral_peak_robust,
        channels=args.channels, fps=args.fps
    )

    r = rng(cfg.seed)
    rows = []
    if cfg.mode == "observational" and cfg.dataset_path:
        rows.append(run_observational(cfg.dataset_path))
    else:
        for t in cfg.treatments:
            rows.append(synth_once(cfg, t, r))

    series = np.array([[row[k] for k in [
        "metric/hr_rmse_bpm","metric/rr_rmse_bpm","metric/hr_mae_bpm"
    ]] for row in rows], dtype=float)
    np.save(os.path.join(args.out_dir, "series.npy"), series)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "per_run_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({k: rows[0][k] for k in rows[0].keys()}, f, indent=2)
