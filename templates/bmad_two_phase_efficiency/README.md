
# AI-Scientist 模板：two_phase_efficiency

## 结构
- `prompt.json`：研究语境、点子生成、实验契约、绘图与论文骨架
- `seed_ideas.json`：起始点子
- `experiment.py`：支持 `simulation` 与 `observational`，写出 `results.json`、`per_run_metrics.csv`、`series.npy`
- `plot.py`：输出 `fig_efficiency.png`、`fig_quality.png`、`fig_sensitivity.png`
- `latex/template.tex`：中文论文模板（ctex）

## 最小运行
```bash
cd templates/two_phase_efficiency
python experiment.py --out_dir run_0 --mode simulation --seed 123   --phase1_investment_ratio 0.4 --review_quality 0.8   --requirement_volatility 0.3 --batch_size 4 --parallelism 4   --engineer_cost_rate 1.0 --context_switch_cost 1.1
python plot.py
```

观测模式（CSV需要包含列）：
`pr_open_ts, pr_merge_ts, is_failed_deploy, restore_hours,
 test_pass_rate, rework_rate, defect_leakage_ppk, review_latency_hours,
 cost_per_feature_unit`

```bash
python experiment.py --out_dir run_obs --mode observational --dataset_path ./telemetry.csv
python plot.py
```
