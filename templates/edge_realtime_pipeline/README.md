
# AI-Scientist 模板：edge_realtime_pipeline

## 结构
- `prompt.json`：研究语境、点子生成、实验契约、绘图与论文骨架
- `seed_ideas.json`：起始点子
- `experiment.py`：支持 `simulation` 与 `observational`，写出 `results.json`、`per_run_metrics.csv`、`series.npy`
- `plot.py`：输出 `fig_latency.png`、`fig_throughput.png`、`fig_resource.png`
- `latex/template.tex`：中文论文模板（ctex）

## 最小运行
```bash
cd templates/edge_realtime_pipeline
python experiment.py --out_dir run_0 --mode simulation --seed 0   --batch_size 4 --window_ms 10 --fusion_level 1 --quant_bits 8   --zero_copy --numa_affinity --rt_priority --cache_policy lru   --warm_pool_size 4 --cold_start_ms 120 --network_rtt_ms 20   --device_class edge_cpu --stream_count 8 --load_factor 1.0
python plot.py
```

观测模式（CSV建议列）：
`lat_p50, lat_p95, lat_p99, jitter, throughput, drop, cpu_util, mem_bw_gbps, energy_mj, cache_hit, cold_rate`
```bash
python experiment.py --out_dir run_obs --mode observational --dataset_path ./edge_telemetry.csv
python plot.py
```
