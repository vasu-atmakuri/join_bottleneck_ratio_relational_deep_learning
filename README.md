# When Does Learned Pooling Fail at Relational Joins?

Code, run records and analysis for the paper *When Does Learned Pooling Fail at Relational Joins? Empirical
Scaling, Architectural Mechanisms, and Benchmark Limits* (V. Atmakuri, AISCN 2027 submission). The manuscript is
in `paper/` (`join_bottleneck_ratio_manuscript.pdf`, LaTeX source `main.tex`).

The study asks when a learned aggregation that compresses the K child rows of a foreign-key join into a
d-dimensional parent embedding stops working, and uses the Join Bottleneck Ratio as an empirical difficulty index:

    JBR = K * H_row / d,        log2 X = log2 K + a_H log2 H_row - a_d log2 d

K is the number of child rows per parent, H_row the summed plug-in entropy of the child columns in bits and d the
embedding width. X is the empirically corrected index (a_H = 0.92, a_d = 0.22 on DeepSets sum).

## Runs at a glance

| Suite | Runner | Records | Completed runs |
|---|---|---|---|
| First synthetic suite (fixed learning rate) | `experiments/run_synthetic_experiments.py` | `results/synthetic_runs/` | 2,202 |
| Learning-rate-tuned core, GNN, extended range, width sweep | `experiments/run_learning_rate_tuned_core_experiments.py` | `results/learning_rate_tuned_core/` | 3,636 |
| Controls (irrelevant columns, missing values, cardinality shift, budget, width) | `experiments/run_capacity_law_control_experiments.py` | `results/capacity_law_controls/` | 1,212 |
| Pre-registered out-of-sample test, two-layer depth | `experiments/run_law_validation_and_depth_experiments.py` | `results/law_validation_and_depth/` | 1,260 |
| New task families, data distributions, query conditioning, attention margins, width budget | `experiments/run_review_robustness_experiments.py` | `results/review_robustness/` | 226 |
| **Controlled total** | | | **8,536** |
| RelBench aggregation and width interventions | `real_data/train_relbench_aggregation_interventions.py` | `results/relbench_interventions/` | 216 |
| RelBench planted labels, attention variants, longer training | `real_data/train_relbench_planted_signal_and_aggregation_variants.py` | `results/relbench_planted_signal/` | 560 |
| **RelBench total** | | | **776** |

`results/review_response_analysis/run_manifest.csv` lists every executed run (suite, experiment, configuration,
target, model, seed, learning rate, status, test metric); `run_manifest_counts.csv` gives the totals above.
Failed RelBench attempts (144 runs with a renamed metric, all rerun; 40 runs on a dropped join) are listed with
status `error`.

## Pre-registration

| Commit | Content |
|---|---|
| `92340b8` | Frozen DeepSets fits and predictions for the 36 out-of-sample configurations (`analysis/preregister_capacity_law_predictions.py`, `results/law_validation_and_depth/preregistered_*`) |
| `85c842e`, `9422a07` | The out-of-sample runs (two seeds, then a third) |
| `f60ed90` | Message-passing predictions and the planted-join inclusion rule (`results/relbench_planted_signal/inclusion_rule.json`) |
| `9d4ad13` | The message-passing out-of-sample runs |

The prediction files have not changed since `92340b8` (`git log -- results/law_validation_and_depth/preregistered_predictions.csv`).
`analysis/analyze_law_validation_and_depth.py` scores them without refitting.

## Layout

| Path | Contents |
|---|---|
| `relational_aggregation_bottleneck/` | data generator and variants, scenario grids, models (DeepSets, one- and two-layer message passing, attention), training loop |
| `experiments/` | resumable runners for the controlled suites |
| `real_data/` | RelBench join statistics, temporal subgraph sampler, aggregation interventions, planted-label experiments, published baseline results |
| `analysis/` | fits, pre-registration, scoring, uncertainty analyses, RelBench analyses, figures (`make_publication_figures.py`) |
| `results/` | one JSON line per run, plus every analysis output used in the paper |
| `paper/` | manuscript source, figures and PDF |

Paper elements and the scripts that produce them:

| Paper | Script | Output |
|---|---|---|
| Fig. 1, Table I | `analysis/analyze_learning_rate_tuned_results.py`, `analysis/analyze_review_response.py` | `results/learning_rate_tuned_analysis/`, `results/review_response_analysis/table1_*` |
| Table II, exponent intervals, baselines | `analysis/analyze_law_validation_and_depth.py`, `analysis/analyze_review_response.py` | `results/law_validation_and_depth/`, `results/review_response_analysis/` |
| Fig. 2, multipliers | `analysis/analyze_law_validation_and_depth.py`, `analysis/analyze_paired_midpoint_shifts.py`, `analysis/analyze_paired_configuration_differences.py` | same folders |
| Table III, query conditioning, attention margins, width | `analysis/analyze_review_robustness.py` | `results/review_robustness/analysis/` |
| Fig. 3, Table IV, per-bin intervals | `analysis/analyze_relbench_interventions_and_planted_signal.py`, `analysis/analyze_planted_bins_with_intervals.py` | `results/relbench_planted_signal/analysis/`, `results/review_response_analysis/planted_bins_with_intervals.csv` |
| RelBench ratio and task difficulty | `analysis/analyze_relbench_jbr_task_difficulty.py`, `analysis/analyze_task_conditional_jbr.py` | `results/relbench_jbr_analysis/`, `results/relbench_task_conditional/` |

## Running

The controlled experiments run on a laptop CPU.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_law_validation_and_depth_experiments.py --workers 3 --threads-per-worker 2
python analysis/analyze_law_validation_and_depth.py
```

Every runner skips finished runs, so it can be interrupted and restarted. The RelBench experiments download the
RelBench databases on first use and build a graph cache under `data/` (not in the repository).
