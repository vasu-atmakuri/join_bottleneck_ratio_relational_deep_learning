# Sources of the published RelBench results

| Column | Source |
|---|---|
| global_median, lightgbm, rdl | Robinson et al., "RelBench: A Benchmark for Deep Learning on Relational Databases", NeurIPS 2024 Datasets and Benchmarks, Tables 3 to 5 (test split). For recommendation, `rdl` is the better of the two RDL variants reported (GraphSAGE or ID-GNN). |
| relgnn | Chen et al., "RelGNN: Composite Message Passing for Relational Deep Learning", ICML 2025 (arXiv:2502.06784), Tables 1 to 3. |
| relgt | Dwivedi et al., "Relational Graph Transformer" (arXiv:2505.10960), Tables 1a and 1b (AUC reported as a fraction there; multiplied by 100 here). Recommendation tasks were not evaluated. |

The LightGBM and RDL values agree between the RelBench and RelGNN papers. All values were re-checked against the arXiv versions of the three papers on 12 November 2026 (RelBench arXiv:2407.20060 Tables 3 to 5; RelGNN arXiv:2502.06784v2 Tables 1 to 3, published at ICML 2025, PMLR 267; RelGT arXiv:2505.10960v2 Tables 1a and 1b, published at ICLR 2026) and match exactly. The RelGT paper reports a re-run RDL baseline that differs from RelBench on rel-stack user-engagement (90.21 against 90.59) and user-badge (89.86 against 88.86); all relative gains here use the RelBench RDL values.
