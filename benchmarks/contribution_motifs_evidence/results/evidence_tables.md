Distinct motifs with evidence / motifs with none, per model:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 25.1 / 3.7 | 47.3 / 14.0 | 59.5 / 8.5 | 44.0 / 16.3 |
| `contribution_motifs()`, `max_seqlets=None` | 28.1 / 4.1 | 52.0 / 12.3 | 60.2 / 8.2 | 40.8 / 18.7 |
| `contribution_motifs()`, one seqlet pass | 8.4 / 0.7 | 33.0 / 0.7 | 41.3 / 2.8 | 40.7 / 10.5 |
| `contribution_motifs()`, observed scores only | 25.8 / 3.6 | 45.3 / 11.0 | 50.7 / 5.8 | 41.2 / 12.5 |
| TF-MoDISco, same regions | 12.4 / 1.4 | 26.7 / 1.0 | 22.2 / 1.8 | 25.5 / 12.8 |
| TF-MoDISco, ENCODE's results | 10.9 / 0.4 | 46.7 / 2.7 | 28.3 / 2.2 | 24.8 / 13.7 |

Share of all runs' distinct motifs with evidence, merged at a cosine of 0.8:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 0.50 | 0.53 | 0.49 | 0.43 |
| `contribution_motifs()`, `max_seqlets=None` | 0.54 | 0.55 | 0.48 | 0.44 |
| `contribution_motifs()`, one seqlet pass | 0.21 | 0.35 | 0.36 | 0.37 |
| `contribution_motifs()`, observed scores only | 0.52 | 0.46 | 0.45 | 0.43 |
| TF-MoDISco, same regions | 0.39 | 0.39 | 0.32 | 0.33 |
| TF-MoDISco, ENCODE's results | 0.34 | 0.57 | 0.38 | 0.34 |

Motifs with outside support / with the model's response only, per model:

| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 10.8 / 15.3 | 27.3 / 21.0 | 32.5 / 27.8 | 30.3 / 14.0 |
| `contribution_motifs()`, `max_seqlets=None` | 11.7 / 17.1 | 26.7 / 25.3 | 30.8 / 30.2 | 28.8 / 12.2 |
| `contribution_motifs()`, one seqlet pass | 5.0 / 3.9 | 21.7 / 11.7 | 23.7 / 18.3 | 28.5 / 12.2 |
| `contribution_motifs()`, observed scores only | 10.9 / 16.0 | 25.3 / 20.7 | 26.8 / 24.2 | 29.7 / 11.8 |
| TF-MoDISco, same regions | 6.4 / 6.1 | 20.0 / 7.0 | 14.3 / 7.8 | 17.2 / 8.3 |
| TF-MoDISco, ENCODE's results | 5.6 / 5.7 | 29.7 / 18.7 | 19.5 / 9.5 | 20.5 / 4.5 |
