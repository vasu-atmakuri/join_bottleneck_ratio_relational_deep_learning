# Join Bottleneck Ratio: Phase Transitions in Relational Aggregation

Code and results for the Join Bottleneck Ratio (JBR) study of information loss when relational deep learning
models aggregate child rows across a foreign-key join into a fixed-size parent embedding.

    JBR = K * H_row / d

K is the number of child rows per parent, H_row the entropy of a child row in bits (sum of per-feature
conditional entropies) and d the embedding dimension of the aggregated parent representation. A
multiset-corrected variant, JBR_set = log2 C(2^H_row + K - 1, K) / d, removes the information in row order,
which a permutation-invariant aggregator cannot use.

## Layout

| Path | Contents |
|---|---|
| `relational_aggregation_bottleneck/scenarios.py` | scenario grids: 25 base (JBR 1/32 to 16, d = 128), 5 iso-JBR, 11 architecture, 38 extreme (JBR 32 to 4096 via smaller d), 5 head-sweep |
| `relational_aggregation_bottleneck/data_generation.py` | parent-child data generator, three targets (mean_based, rare_pattern, distributional), empirical JBR |
| `relational_aggregation_bottleneck/models.py` | DeepSets (sum, mean, max), GNN SumAgg / MeanAgg / AttnAgg (multi-head, parent query), denormalized MLP |
| `relational_aggregation_bottleneck/training.py` | online training (fresh sample per step), early stopping on validation AUROC |
| `experiments/run_synthetic_experiments.py` | resumable runner for all 2,202 synthetic training runs |
| `analysis/analyze_synthetic_phase_transition.py` | sigmoid fits, predictor comparison, regimes, iso-JBR, architecture, capacity multiplier, figures |
| `results/synthetic_runs/` | one JSON line per training run |
| `results/synthetic_analysis/` | tables, summary JSON and figures |

## Synthetic data

Each parent has a query pattern q in {0,1}^D and K child rows of D uniform binary features.

* **mean_based**: 1 if w . (mean child row - 1/2) > 0 for a fixed random w. Needs only first moments.
* **rare_pattern**: 1 if exactly one child equals q, 0 if none does.
* **distributional**: 1 if q appears exactly twice among the children, 0 if exactly once.

Non-planted rows never equal q, so the classes of the two set targets differ only in the planted rows.
Training batches are generated fresh at every step, so results measure the aggregation channel rather than
the size of a fixed training set. Validation (1,000 parents) and test (2,000 parents) sets are fixed.

## Experiments

| Experiment | Runs |
|---|---|
| main_deepsets: 25 scenarios x 3 targets x {sum, mean, max} x 3 seeds | 675 |
| gnn_architecture: 11 scenarios x 3 targets x {SumAgg, MeanAgg, AttnAgg} x 3 seeds | 297 |
| attention_head_sweep: 5 scenarios x {1, 4, 8, 16} heads x 3 seeds | 60 |
| extended_attention: 63 scenarios x {1, 4, 8, 16} heads x 3 seeds | 756 |
| extended_sum: 63 scenarios x SumAgg x 3 seeds | 189 |
| denormalized: 25 scenarios x 3 targets x 3 seeds | 225 |
| **total** | **2,202** |

## Running

Runs on a laptop CPU (developed on an 8 GB Apple M1). Three worker processes with two threads each use
about 1.5 GB of memory in total.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_synthetic_experiments.py --pilot --results results/pilot_runs.jsonl
python experiments/run_synthetic_experiments.py --workers 3 --threads-per-worker 2
python analysis/analyze_synthetic_phase_transition.py
```

The runner skips finished runs, so it can be interrupted and restarted at any time.
