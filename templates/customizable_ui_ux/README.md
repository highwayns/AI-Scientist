
# AI-Scientist 模板：customizable_ui_ux

## 结构
- `prompt.json`：研究语境、点子生成、实验契约、绘图与论文骨架
- `seed_ideas.json`：起始点子
- `experiment.py`：支持 `simulation` 与 `observational`，写出 `results.json`、`per_run_metrics.csv`、`series.npy`
- `plot.py`：输出 `fig_main.png`、`fig_quality.png`、`fig_latency.png`
- `latex/template.tex`：中文论文模板（ctex）

## 最小运行
```bash
cd templates/customizable_ui_ux
python experiment.py --out_dir run_0 --mode simulation --seed 0   --personalization_depth 0.7 --latency_budget_ms 150   --accessibility_contrast 0.7 --font_scale 1.2   --layout_density 1.0 --device_class mobile   --input_mix_touch_ratio 0.7 --context_switch_rate 0.4
python plot.py
```

观测模式（CSV建议列）：
`success, completion_time_s, error, sus, tlx, scroll_depth_ratio, viewport_utilization, interaction_latency_ms, a11y_events`
```bash
python experiment.py --out_dir run_obs --mode observational --dataset_path ./usability.csv
python plot.py
```
