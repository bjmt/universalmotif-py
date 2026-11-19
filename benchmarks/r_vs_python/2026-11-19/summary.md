| Task | Size | Threads | R | Python | Speed-up | R memory | Python memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Read MEME motifs | 8,790 motifs | 1 | 1.41 s | 0.0351 s | 40x | 26 MB | 22 MB |
| Write MEME motifs | 8,790 motifs | 1 | 1.25 s | 0.039 s | 32x | 103 MB | 21 MB |
| Convert motifs to PWMs | 87,900 motifs | 1 | 5.84 s | 0.113 s | 52x | 203 MB | 81 MB |
| Create a motif from sites | 2,000,000 sites | 1 | 1.6 s | 0.35 s | 5x | 1,055 MB | 306 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 1 | 35.5 s | 0.308 s | 115x | 23 MB | 12 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 2 | 18.2 s | 0.166 s | 110x | 21 MB | 13 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 1 | 35.4 s | 0.494 s | 72x | 29 MB | 105 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 2 | 18 s | 0.297 s | 61x | 20 MB | 106 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 1 | 24 s | 0.215 s | 112x | 142 MB | 27 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 2 | 12.3 s | 0.12 s | 103x | 140 MB | 24 MB |
| Enrichment of 100 motifs | 1,000 peaks | 1 | 34.8 s | 0.303 s | 115x | 85 MB | 39 MB |
| Enrichment of 100 motifs | 1,000 peaks | 2 | 17.9 s | 0.168 s | 107x | 81 MB | 36 MB |
| Score thresholds for 100 P-values | 879 motifs | 1 | 13.4 s | 0.553 s | 24x | 32 MB | 36 MB |
| Score thresholds for 100 P-values | 879 motifs | 2 | 6.88 s | 0.326 s | 21x | 28 MB | 46 MB |
| Compare all against all, with P-values | 879 motifs | 1 | 47.2 s | 2.05 s | 23x | 391 MB | 291 MB |
| Compare all against all, with P-values | 879 motifs | 2 | 25 s | 1.18 s | 21x | 367 MB | 292 MB |
| Merge similar motifs | 879 motifs | 1 | 47.2 s | 2.02 s | 23x | 392 MB | 272 MB |
| Merge similar motifs | 879 motifs | 2 | 25.2 s | 1.14 s | 22x | 345 MB | 274 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 1 | 3.37 s | 0.182 s | 18x | 202 MB | 172 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 2 | 2.01 s | 0.113 s | 18x | 208 MB | 118 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 1 | 10.3 s | 0.641 s | 16x | 215 MB | 116 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 2 | 5.57 s | 0.444 s | 13x | 215 MB | 119 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 1 | 14.3 s | 0.626 s | 23x | 801 MB | 102 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 2 | 11.7 s | 0.389 s | 30x | 820 MB | 102 MB |
| Sequence complexity | 100,000 500 bp sequences | 1 | 24 s | 0.715 s | 34x | 2,623 MB | 590 MB |
| Sequence complexity | 100,000 500 bp sequences | 2 | 21.3 s | 0.504 s | 42x | 2,697 MB | 590 MB |
