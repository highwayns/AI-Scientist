
# AI-Scientist Template: bmad_agent_agile

## Structure
- `prompt.json`: research context, idea generation, experiment contract, plotting, paper structure
- `seed_ideas.json`: starter ideas for LLM
- `experiment.py`: supports `simulation` and `observational` modes; writes `results.json`, `per_run_metrics.csv`, `series.npy`
- `plot.py`: produces `fig_main.png`, `fig_ablation.png`, `fig_network.png`
- `latex/template.tex`: minimal paper skeleton

## Quick Start
```bash
cd templates/bmad_agent_agile
python experiment.py --out_dir run_0 --mode simulation --seed 7
python plot.py
```

For observational mode (requires a CSV with columns):
`pr_open_ts, pr_merge_ts, is_failed_deploy, restore_hours, reviewer_role, author_role, files_roles`

```bash
python experiment.py --out_dir run_obs --mode observational --dataset_path ./telemetry.csv
python plot.py
```
