
# AI-Scientist 模板：bmad_task_handoff_ai

## 结构
- `prompt.json`：研究语境、点子生成、实验契约、绘图与论文骨架
- `seed_ideas.json`：起始点子
- `experiment.py`：支持 `simulation` 与 `observational`，写出 `results.json`、`per_run_metrics.csv`、`series.npy`
- `plot.py`：输出 `fig_main.png`、`fig_ablation.png`、`fig_handoff.png`
- `latex/template.tex`：中文论文模板（ctex）

## 最小运行
```bash
cd templates/bmad_task_handoff_ai
python experiment.py --out_dir run_0 --mode simulation --seed 7   --task_granularity 1.0 --handoff_package_completeness 0.8   --rag_quality 0.8 --llm_window_tokens 32000 --handoff_chain_length 3

python plot.py
```

观测模式（CSV需要包含列）：
`pr_open_ts, pr_merge_ts, is_failed_deploy, restore_hours, reviewer_role, author_role, files_roles, handoff_chain_length, context_missing_fields, context_total_fields, tokens_used`

```bash
python experiment.py --out_dir run_obs --mode observational --dataset_path ./telemetry.csv
python plot.py
```
