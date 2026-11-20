Distinct motifs with evidence / motifs with none, per model:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 21.1 / 7.8 | 39.3 / 22.0 | 37.8 / 30.3 | 35.5 / 24.8 |
| `contribution_motifs()`, `max_seqlets=None` | 23.8 / 8.4 | 41.7 / 22.7 | 37.7 / 30.7 | 34.5 / 25.0 |
| `contribution_motifs()`, one seqlet pass | 8.4 / 0.7 | 32.0 / 1.7 | 32.8 / 11.3 | 36.0 / 15.2 |
| `contribution_motifs()`, observed scores only | 21.8 / 7.6 | 37.7 / 18.7 | 31.5 / 25.0 | 33.3 / 20.3 |
| TF-MoDISco, same regions | 11.6 / 2.2 | 26.7 / 1.0 | 17.0 / 7.0 | 21.2 / 17.2 |
| TF-MoDISco, ENCODE's results | 10.6 / 0.8 | 46.0 / 3.7 | 23.0 / 7.5 | 22.5 / 16.0 |

Cells that differ under the IC-matched rule: TF-MoDISco, same regions, ProCapNet profile, neither: 17.17 (width) against 15.17 (information content).

Share of all runs' distinct motifs with evidence, merged at a cosine of 0.8:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 0.53 | 0.51 | 0.52 | 0.44 |
| `contribution_motifs()`, `max_seqlets=None` | 0.58 | 0.53 | 0.53 | 0.48 |
| `contribution_motifs()`, one seqlet pass | 0.26 | 0.40 | 0.46 | 0.41 |
| `contribution_motifs()`, observed scores only | 0.50 | 0.47 | 0.50 | 0.44 |
| TF-MoDISco, same regions | 0.43 | 0.44 | 0.39 | 0.35 |
| TF-MoDISco, ENCODE's results | 0.41 | 0.64 | 0.52 | 0.39 |

Cells that differ under the IC-matched rule: `contribution_motifs()`, BPNet, share_of_union: 0.53 (width) against 0.46 (information content); `contribution_motifs()`, ProCapNet counts, share_of_union: 0.52 (width) against 0.55 (information content); `contribution_motifs()`, ProCapNet profile, share_of_union: 0.44 (width) against 0.42 (information content); `contribution_motifs()`, `max_seqlets=None`, BPNet, share_of_union: 0.58 (width) against 0.51 (information content); `contribution_motifs()`, one seqlet pass, BPNet, share_of_union: 0.26 (width) against 0.23 (information content); `contribution_motifs()`, observed scores only, BPNet, share_of_union: 0.50 (width) against 0.47 (information content).

Motifs with outside support / with the model's response only, per model:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 10.8 / 11.2 | 27.7 / 12.7 | 32.3 / 6.2 | 30.0 / 5.8 |
| `contribution_motifs()`, `max_seqlets=None` | 11.7 / 12.8 | 26.7 / 15.0 | 31.7 / 6.8 | 29.7 / 5.0 |
| `contribution_motifs()`, one seqlet pass | 4.9 / 3.9 | 21.0 / 11.3 | 24.3 / 9.2 | 29.0 / 7.0 |
| `contribution_motifs()`, observed scores only | 10.9 / 12.1 | 25.7 / 12.7 | 27.2 / 4.7 | 30.0 / 3.7 |
| TF-MoDISco, same regions | 6.5 / 5.2 | 19.3 / 7.7 | 14.7 / 2.3 | 16.7 / 4.5 |
| TF-MoDISco, ENCODE's results | 5.6 / 5.3 | 30.0 / 17.3 | 19.7 / 4.0 | 19.8 / 2.8 |

Cells that differ under the IC-matched rule: none.

Distinct motifs with evidence / motifs with none, per model, under the IC-matched rule:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 21.2 / 7.7 | 39.0 / 22.3 | 38.2 / 30.0 | 35.8 / 24.5 |
| `contribution_motifs()`, `max_seqlets=None` | 24.2 / 7.9 | 40.3 / 24.0 | 36.8 / 31.5 | 35.2 / 24.3 |
| `contribution_motifs()`, one seqlet pass | 8.2 / 0.9 | 32.0 / 1.7 | 31.7 / 12.5 | 37.2 / 14.0 |
| `contribution_motifs()`, observed scores only | 22.2 / 7.1 | 38.3 / 18.0 | 31.7 / 24.8 | 34.8 / 18.8 |
| TF-MoDISco, same regions | 12.0 / 1.8 | 26.7 / 1.0 | 17.5 / 6.5 | 23.2 / 15.2 |
| TF-MoDISco, ENCODE's results | 10.8 / 0.6 | 45.7 / 4.0 | 23.2 / 7.3 | 23.8 / 14.7 |
