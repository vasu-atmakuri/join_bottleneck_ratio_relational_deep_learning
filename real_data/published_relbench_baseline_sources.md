# Sources of the published RelBench results

| Column | Source |
|---|---|
| global_median, lightgbm, rdl | Robinson et al., "RelBench: A Benchmark for Deep Learning on Relational Databases", NeurIPS 2024 Datasets and Benchmarks, Tables 3 to 5 (test split). For recommendation, `rdl` is the better of the two RDL variants reported (GraphSAGE or ID-GNN). |
| relgnn | Chen et al., "RelGNN: Composite Message Passing for Relational Deep Learning", ICML 2025 (arXiv:2502.06784), Tables 1 to 3. |
| relgt | Dwivedi et al., "Relational Graph Transformer" (arXiv:2505.10960), Tables 1a and 1b (AUC reported as a fraction there; multiplied by 100 here). Recommendation tasks were not evaluated. |

The LightGBM and RDL values agree between the RelBench and RelGNN papers. Values were transcribed from the papers and must be re-checked against the PDFs before publication.
