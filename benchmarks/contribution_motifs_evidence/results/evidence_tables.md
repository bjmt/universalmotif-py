Distinct motifs with evidence / motifs with none, per model:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 21.9 / 6.8 | 37.3 / 24.0 | 41.5 / 26.7 | 36.0 / 24.3 |
| `contribution_motifs()`, `max_seqlets=None` | 24.4 / 7.8 | 42.0 / 22.3 | 40.0 / 28.3 | 35.7 / 23.8 |
| `contribution_motifs()`, one seqlet pass | 8.4 / 0.7 | 32.0 / 1.7 | 34.2 / 10.0 | 35.8 / 15.3 |
| `contribution_motifs()`, observed scores only | 22.1 / 7.2 | 36.7 / 19.7 | 35.7 / 20.8 | 34.8 / 18.8 |
| TF-MoDISco, same regions | 11.4 / 2.4 | 26.7 / 1.0 | 17.5 / 6.5 | 22.2 / 16.2 |
| TF-MoDISco, ENCODE's results | 10.4 / 1.0 | 45.3 / 4.0 | 24.0 / 6.5 | 22.0 / 16.5 |

Cells that differ under the IC-matched rule: TF-MoDISco, same regions, ProCapNet profile, neither: 16.17 (width) against 14.17 (information content).

Share of all runs' distinct motifs with evidence, merged at a cosine of 0.8:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 0.52 | 0.49 | 0.52 | 0.44 |
| `contribution_motifs()`, `max_seqlets=None` | 0.54 | 0.56 | 0.53 | 0.50 |
| `contribution_motifs()`, one seqlet pass | 0.25 | 0.39 | 0.46 | 0.40 |
| `contribution_motifs()`, observed scores only | 0.50 | 0.46 | 0.49 | 0.46 |
| TF-MoDISco, same regions | 0.44 | 0.45 | 0.39 | 0.36 |
| TF-MoDISco, ENCODE's results | 0.40 | 0.64 | 0.49 | 0.37 |

Cells that differ under the IC-matched rule: `contribution_motifs()`, BPNet, share_of_union: 0.52 (width) against 0.49 (information content); `contribution_motifs()`, ProCapNet profile, share_of_union: 0.44 (width) against 0.41 (information content); `contribution_motifs()`, `max_seqlets=None`, BPNet, share_of_union: 0.54 (width) against 0.51 (information content); `contribution_motifs()`, `max_seqlets=None`, ProCapNet counts, share_of_union: 0.53 (width) against 0.50 (information content); `contribution_motifs()`, `max_seqlets=None`, ProCapNet profile, share_of_union: 0.50 (width) against 0.47 (information content); `contribution_motifs()`, one seqlet pass, ProCapNet counts, share_of_union: 0.46 (width) against 0.44 (information content); `contribution_motifs()`, observed scores only, BPNet, share_of_union: 0.50 (width) against 0.48 (information content); `contribution_motifs()`, observed scores only, ProCapNet profile, share_of_union: 0.46 (width) against 0.44 (information content); TF-MoDISco, same regions, BPNet, share_of_union: 0.44 (width) against 0.46 (information content).

Motifs with outside support / with the model's response only, per model:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 10.8 / 12.1 | 27.7 / 10.7 | 32.3 / 9.8 | 30.0 / 6.3 |
| `contribution_motifs()`, `max_seqlets=None` | 11.7 / 13.4 | 26.7 / 15.3 | 31.7 / 9.2 | 29.7 / 6.2 |
| `contribution_motifs()`, one seqlet pass | 4.9 / 3.9 | 21.0 / 11.3 | 24.3 / 10.5 | 29.0 / 6.8 |
| `contribution_motifs()`, observed scores only | 10.9 / 12.4 | 25.7 / 11.7 | 27.2 / 8.8 | 30.0 / 5.2 |
| TF-MoDISco, same regions | 6.5 / 5.1 | 19.3 / 7.7 | 14.7 / 2.8 | 16.7 / 5.5 |
| TF-MoDISco, ENCODE's results | 5.6 / 5.1 | 30.0 / 17.0 | 19.7 / 5.0 | 19.8 / 2.3 |

Cells that differ under the IC-matched rule: none.

Distinct motifs with evidence / motifs with none, per model, under the IC-matched rule:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 21.4 / 7.3 | 37.7 / 23.7 | 42.8 / 25.2 | 35.7 / 24.7 |
| `contribution_motifs()`, `max_seqlets=None` | 23.8 / 8.4 | 43.0 / 21.3 | 40.3 / 28.0 | 35.5 / 24.0 |
| `contribution_motifs()`, one seqlet pass | 8.2 / 0.9 | 32.3 / 1.3 | 34.8 / 9.3 | 37.0 / 14.2 |
| `contribution_motifs()`, observed scores only | 21.6 / 7.8 | 36.7 / 19.7 | 36.7 / 19.8 | 34.8 / 18.8 |
| TF-MoDISco, same regions | 12.3 / 1.5 | 26.7 / 1.0 | 18.7 / 5.3 | 24.2 / 14.2 |
| TF-MoDISco, ENCODE's results | 10.8 / 0.6 | 46.0 / 3.7 | 25.3 / 5.2 | 23.3 / 15.2 |
