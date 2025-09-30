
# AI-Scientist 模板：contactless_vitals_robustness

## 结构
- `prompt.json`：研究语境、点子生成、实验契约、绘图与论文骨架
- `seed_ideas.json`：起始点子
- `experiment.py`：支持 `simulation` 与 `observational`，写出 `results.json`、`per_run_metrics.csv`、`series.npy`
- `plot.py`：输出 `fig_acc.png`、`fig_robust.png`、`fig_ba.png`
- `latex/template.tex`：中文论文模板（ctex）

## 最小运行
```bash
cd templates/contactless_vitals_robustness
python experiment.py --out_dir run_0 --mode simulation --seed 0   --modality rgb --distance_m 1.5 --illum_lux 80 --motion_level 0.3   --skin_tone 0.6 --occlusion_ratio 0.1 --wqm_threshold 0.6   --roi_adapt_strength 0.8 --temporal_smooth_strength 0.4 --spectral_peak_robust   --channels 3 --fps 30
python plot.py
```

观测模式（CSV建议列）：
`ref_hr, ref_rr, pred_hr, pred_rr, snr_db, latency_ms, power_mw, modality, skin_tone, distance_m, illum_lux, motion_level, occlusion_ratio, sdnn_bias`
```bash
python experiment.py --out_dir run_obs --mode observational --dataset_path ./vitals.csv
python plot.py
```
