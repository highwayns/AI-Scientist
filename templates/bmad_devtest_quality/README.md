
# AI-Scientist 模板：bmad_devtest_quality

## 结构
- `prompt.json`：研究语境、点子生成、实验契约、绘图与论文骨架
- `seed_ideas.json`：起始点子
- `experiment.py`：支持 `simulation` 与 `observational`，写出 `results.json`、`per_run_metrics.csv`、`series.npy`
- `plot.py`：输出 `fig_main.png`、`fig_quality.png`、`fig_cost.png`
- `latex/template.tex`：中文论文模板（ctex）

## 最小运行
```bash
cd templates/bmad_devtest_quality
python experiment.py --out_dir run_0 --mode simulation --seed 42   --quality_gate_threshold 0.75 --ai_test_gen_coverage 0.7   --test_coverage_target 0.8 --mutation_score_target 0.75   --flaky_rate 0.1 --parallelism 4 --pipeline_caching 0.5   --change_size_loc 400 --risk_profile 0.5

python plot.py
```

观测模式（CSV需要包含列）：
`pr_open_ts, pr_merge_ts, is_failed_deploy, restore_hours, test_pass_rate, test_coverage, mutation_score, flaky_fail_share, pipeline_duration_min, pipeline_cost_unit, defect_leakage_ppk`

```bash
python experiment.py --out_dir run_obs --mode observational --dataset_path ./telemetry.csv
python plot.py
```
