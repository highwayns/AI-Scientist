
# AI-Scientist 模板：model_drift_xai

## 结构
- `prompt.json`：研究语境、点子生成、实验契约、绘图与论文骨架
- `seed_ideas.json`：起始点子
- `experiment.py`：支持 `simulation` 与 `observational`，写出 `results.json`、`per_run_metrics.csv`、`series.npy`
- `plot.py`：输出 `fig_perf.png`、`fig_drift.png`、`fig_xai.png`
- `latex/template.tex`：中文论文模板（ctex）

## 最小运行
```bash
cd templates/model_drift_xai
python experiment.py --out_dir run_0 --mode simulation --seed 0   --drift_type covariate --drift_magnitude 0.4 --window_size 1000 --alert_threshold 0.1
python plot.py
```

观测模式（目录需包含 pre.csv 与 post.csv，列：`y, pred_prob, f1, f2, ...`）
```bash
python experiment.py --out_dir run_obs --mode observational --dataset_ref ./dataset_dir
python plot.py
```
