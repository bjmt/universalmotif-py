Each cell is given with its mean over the bootstraps (in parentheses) and their 5th to 95th percentiles. Mean less cell, median and largest:

distinct_evidenced -0.09, -0.79, with 0 of 24 cells outside their interval;
neither +0.10, +0.79, with 0 of 24 cells outside their interval;
model_only -0.10, -0.79, with 0 of 24 cells outside their interval;
share_of_union -0.00, -0.03, with 0 of 24 cells outside their interval.

Distinct motifs with evidence / motifs with none, per model:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 20.4 (19.6) [18.6, 20.8] / 8.4 (9.2) [8.1, 10.3] | 36.7 (36.8) [35.3, 38.3] / 24.7 (24.5) [23.0, 26.0] | 36.2 (36.6) [35.0, 38.3] / 32.0 (31.5) [29.8, 33.2] | 34.0 (33.8) [32.8, 35.2] / 26.3 (26.5) [25.2, 27.5] |
| `contribution_motifs()`, `max_seqlets=None` | 22.8 (22.1) [20.9, 23.2] / 9.4 (10.1) [9.0, 11.3] | 40.3 (39.5) [37.3, 41.7] / 24.0 (24.8) [22.7, 27.0] | 35.2 (35.8) [34.0, 37.7] / 33.2 (32.5) [30.7, 34.3] | 34.2 (33.8) [32.7, 34.8] / 25.3 (25.7) [24.7, 26.8] |
| `contribution_motifs()`, one seqlet pass | 8.2 (8.2) [8.0, 8.4] / 0.9 (0.9) [0.7, 1.1] | 31.7 (31.8) [31.0, 32.7] / 2.0 (1.9) [1.0, 2.7] | 30.5 (30.4) [28.7, 32.0] / 13.7 (13.8) [12.2, 15.5] | 34.2 (34.0) [32.8, 35.3] / 17.0 (17.2) [15.8, 18.3] |
| `contribution_motifs()`, observed scores only | 21.1 (20.4) [19.1, 21.4] / 8.2 (8.9) [7.9, 10.2] | 35.0 (35.5) [34.0, 37.3] / 21.3 (20.8) [19.0, 22.3] | 31.0 (30.9) [29.7, 32.7] / 25.5 (25.6) [23.8, 26.8] | 33.7 (33.3) [32.3, 34.5] / 20.0 (20.3) [19.2, 21.3] |
| TF-MoDISco, same regions | 11.2 (11.1) [10.8, 11.4] / 2.6 (2.7) [2.4, 3.1] | 26.3 (26.4) [26.0, 26.7] / 1.3 (1.3) [1.0, 1.7] | 16.3 (16.6) [16.0, 17.3] / 7.7 (7.4) [6.7, 8.0] | 20.5 (20.2) [19.5, 20.8] / 17.8 (18.1) [17.5, 18.8] |
| TF-MoDISco, ENCODE's results | 10.2 (10.2) [9.9, 10.4] / 1.2 (1.2) [1.0, 1.4] | 45.3 (44.9) [44.0, 45.7] / 4.3 (4.7) [3.7, 5.7] | 22.2 (22.1) [21.5, 22.8] / 8.5 (8.4) [7.7, 9.2] | 22.0 (22.0) [21.5, 22.3] / 16.5 (16.5) [16.2, 17.0] |

Each run's distinct_evidenced less that of `contribution_motifs()`, with its bootstrap interval (* where the interval holds 0):

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()`, `max_seqlets=None` | +2.3 [+1.8, +3.0] | +3.7 [+1.0, +4.3] | -1.0 [-1.8, +0.2]* | +0.2 [-0.8, +0.8]* |
| `contribution_motifs()`, one seqlet pass | -12.2 [-12.5, -10.4] | -5.0 [-6.7, -3.7] | -5.7 [-7.3, -5.5] | +0.2 [-0.7, +1.2]* |
| `contribution_motifs()`, observed scores only | +0.6 [+0.2, +1.4] | -1.7 [-2.7, +0.3]* | -5.2 [-6.7, -4.8] | -0.3 [-1.5, +0.3]* |
| TF-MoDISco, same regions | -9.2 [-9.7, -7.5] | -10.3 [-12.0, -9.0] | -19.8 [-21.8, -18.5] | -13.5 [-15.0, -12.5] |
| TF-MoDISco, ENCODE's results | -10.2 [-10.5, -8.4] | +8.7 [+6.7, +9.7] | -14.0 [-15.8, -13.3] | -12.0 [-13.0, -10.8] |

Share of all runs' distinct motifs with evidence, merged at a cosine of 0.8:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 0.53 (0.50) [0.48, 0.54] | 0.49 (0.49) [0.47, 0.51] | 0.55 (0.55) [0.54, 0.57] | 0.44 (0.44) [0.43, 0.46] |
| `contribution_motifs()`, `max_seqlets=None` | 0.57 (0.54) [0.52, 0.57] | 0.55 (0.54) [0.52, 0.56] | 0.55 (0.54) [0.53, 0.56] | 0.51 (0.51) [0.50, 0.53] |
| `contribution_motifs()`, one seqlet pass | 0.26 (0.27) [0.26, 0.28] | 0.40 (0.40) [0.39, 0.42] | 0.47 (0.47) [0.46, 0.48] | 0.41 (0.42) [0.40, 0.43] |
| `contribution_motifs()`, observed scores only | 0.51 (0.50) [0.49, 0.52] | 0.47 (0.47) [0.45, 0.48] | 0.52 (0.52) [0.51, 0.53] | 0.48 (0.48) [0.47, 0.49] |
| TF-MoDISco, same regions | 0.45 (0.45) [0.43, 0.46] | 0.45 (0.45) [0.44, 0.46] | 0.40 (0.41) [0.39, 0.42] | 0.37 (0.36) [0.35, 0.38] |
| TF-MoDISco, ENCODE's results | 0.41 (0.42) [0.41, 0.44] | 0.65 (0.64) [0.63, 0.66] | 0.53 (0.53) [0.51, 0.54] | 0.40 (0.40) [0.39, 0.42] |

Each run's share_of_union less that of `contribution_motifs()`, with its bootstrap interval (* where the interval holds 0):

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()`, `max_seqlets=None` | +0.04 [+0.02, +0.05] | +0.06 [+0.02, +0.08] | -0.00 [-0.03, +0.01]* | +0.07 [+0.06, +0.08] |
| `contribution_motifs()`, one seqlet pass | -0.27 [-0.27, -0.21] | -0.09 [-0.11, -0.06] | -0.08 [-0.10, -0.07] | -0.03 [-0.04, -0.01] |
| `contribution_motifs()`, observed scores only | -0.02 [-0.03, +0.03]* | -0.02 [-0.05, +0.00]* | -0.03 [-0.05, -0.02] | +0.04 [+0.01, +0.05] |
| TF-MoDISco, same regions | -0.09 [-0.10, -0.02] | -0.04 [-0.06, -0.01] | -0.15 [-0.16, -0.13] | -0.07 [-0.10, -0.06] |
| TF-MoDISco, ENCODE's results | -0.12 [-0.12, -0.05] | +0.16 [+0.13, +0.18] | -0.02 [-0.04, -0.01] | -0.04 [-0.06, -0.03] |

Motifs with outside support / with the model's response only, per model:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 10.8 / 10.6 (9.8) [8.7, 10.9] | 27.7 / 10.0 (10.2) [8.7, 11.7] | 32.3 / 4.5 (5.0) [3.3, 6.7] | 30.0 / 4.3 (4.1) [3.2, 5.5] |
| `contribution_motifs()`, `max_seqlets=None` | 11.7 / 11.8 (11.2) [10.0, 12.2] | 26.7 / 13.7 (12.9) [10.7, 15.0] | 31.7 / 4.3 (5.0) [3.2, 6.8] | 29.7 / 4.7 (4.3) [3.2, 5.3] |
| `contribution_motifs()`, one seqlet pass | 4.9 / 3.8 (3.7) [3.5, 3.9] | 21.0 / 11.0 (11.1) [10.3, 12.0] | 24.3 / 6.8 (6.7) [5.0, 8.3] | 29.0 / 5.2 (5.0) [3.8, 6.3] |
| `contribution_motifs()`, observed scores only | 10.9 / 11.4 (10.7) [9.4, 11.8] | 25.7 / 10.0 (10.5) [9.0, 12.3] | 27.2 / 4.2 (4.1) [2.8, 5.8] | 30.0 / 4.0 (3.7) [2.7, 4.8] |
| TF-MoDISco, same regions | 6.5 / 4.9 (4.8) [4.4, 5.1] | 19.3 / 7.3 (7.4) [7.0, 7.7] | 14.7 / 1.7 (1.9) [1.3, 2.7] | 16.7 / 3.8 (3.6) [2.8, 4.2] |
| TF-MoDISco, ENCODE's results | 5.6 / 4.9 (4.9) [4.7, 5.1] | 30.0 / 16.7 (16.3) [15.3, 17.3] | 19.7 / 3.0 (3.1) [2.3, 3.8] | 19.8 / 2.3 (2.3) [1.8, 2.7] |

Distinct motifs with evidence / motifs with none, per model, under the IC-matched rule (without intervals):

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 20.4 / 8.4 | 37.3 / 24.0 | 35.8 / 32.3 | 33.8 / 26.5 |
| `contribution_motifs()`, `max_seqlets=None` | 22.8 / 9.4 | 40.3 / 24.0 | 35.5 / 32.8 | 34.0 / 25.5 |
| `contribution_motifs()`, one seqlet pass | 8.3 / 0.8 | 32.0 / 1.7 | 29.2 / 15.0 | 34.7 / 16.5 |
| `contribution_motifs()`, observed scores only | 20.9 / 8.4 | 36.0 / 20.3 | 30.5 / 26.0 | 33.7 / 20.0 |
| TF-MoDISco, same regions | 12.1 / 1.7 | 26.3 / 1.3 | 17.2 / 6.8 | 23.2 / 15.2 |
| TF-MoDISco, ENCODE's results | 10.8 / 0.6 | 45.7 / 4.0 | 22.5 / 8.2 | 23.7 / 14.8 |

