
# AI-Scientist 模板：blockchain_audit_immutable

## 结构
- `prompt.json`：研究语境、点子生成、实验契约、绘图与论文骨架
- `seed_ideas.json`：起始点子
- `experiment.py`：支持 `simulation` 与 `observational`，写出 `results.json`、`per_run_metrics.csv`、`series.npy`
- `plot.py`：输出 `fig_security.png`、`fig_cost_latency.png`、`fig_tradeoff.png`
- `latex/template.tex`：中文论文模板（ctex）

## 最小运行
```bash
cd templates/blockchain_audit_immutable
python experiment.py --out_dir run_0 --mode simulation --seed 0   --anchor_frequency_sec 60 --batch_size 50 --chain_confirmations 3   --tx_cost_unit 1.0 --reorg_prob 0.005 --adversary_strength 0.4
python plot.py
```

观测模式（CSV建议列）：
`timestamp, is_tamper_attempt, detected, is_false_positive, anchoring_latency_sec, anchoring_cost_unit, storage_overhead_ratio, incident_mttd_hours, legal_hold_success_rate, throughput_logs_per_sec`
```bash
python experiment.py --out_dir run_obs --mode observational --dataset_path ./audit_logs.csv
python plot.py
```
