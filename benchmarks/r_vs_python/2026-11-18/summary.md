| Task | Size | Threads | R | Python | Speed-up | R memory | Python memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Read MEME motifs | 8,790 motifs | 1 | 1.37 s | 0.0352 s | 39x | 27 MB | 22 MB |
| Write MEME motifs | 8,790 motifs | 1 | 1.23 s | 0.0364 s | 34x | 99 MB | 20 MB |
| Convert motifs to PWMs | 87,900 motifs | 1 | 6.05 s | 0.125 s | 49x | 211 MB | 81 MB |
| Create a motif from sites | 2,000,000 sites | 1 | 1.61 s | 0.356 s | 5x | 1,059 MB | 306 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 1 | 36.3 s | 0.384 s | 94x | 22 MB | 11 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 2 | 18.4 s | 0.208 s | 88x | 22 MB | 13 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 1 | 35.8 s | 0.555 s | 64x | 22 MB | 47 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 2 | 18 s | 0.332 s | 54x | 22 MB | 108 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 1 | 24.5 s | 0.272 s | 90x | 138 MB | 27 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 2 | 12.4 s | 0.148 s | 84x | 132 MB | 24 MB |
| Enrichment of 100 motifs | 1,000 peaks | 1 | 35.6 s | 0.356 s | 100x | 77 MB | 37 MB |
| Enrichment of 100 motifs | 1,000 peaks | 2 | 18 s | 0.197 s | 91x | 83 MB | 36 MB |
| Score thresholds for 100 P-values | 879 motifs | 1 | 13.9 s | 0.585 s | 24x | 31 MB | 38 MB |
| Score thresholds for 100 P-values | 879 motifs | 2 | 6.92 s | 0.328 s | 21x | 35 MB | 47 MB |
| Compare all against all, with P-values | 879 motifs | 1 | 48.4 s | 2.07 s | 23x | 391 MB | 291 MB |
| Compare all against all, with P-values | 879 motifs | 2 | 25.3 s | 1.22 s | 21x | 383 MB | 293 MB |
| Merge similar motifs | 879 motifs | 1 | 48.6 s | 2.06 s | 24x | 398 MB | 273 MB |
| Merge similar motifs | 879 motifs | 2 | 25.5 s | 1.17 s | 22x | 393 MB | 274 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 1 | 3.63 s | 0.196 s | 19x | 200 MB | 115 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 2 | 1.99 s | 0.116 s | 17x | 209 MB | 116 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 1 | 10.6 s | 0.678 s | 16x | 229 MB | 116 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 2 | 5.69 s | 0.45 s | 13x | 219 MB | 117 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 1 | 14.7 s | 0.639 s | 23x | 814 MB | 103 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 2 | 12 s | 0.404 s | 30x | 800 MB | 104 MB |
| Sequence complexity | 100,000 500 bp sequences | 1 | 24.6 s | 0.738 s | 33x | 2,616 MB | 649 MB |
| Sequence complexity | 100,000 500 bp sequences | 2 | 21.7 s | 0.55 s | 39x | 2,708 MB | 591 MB |
