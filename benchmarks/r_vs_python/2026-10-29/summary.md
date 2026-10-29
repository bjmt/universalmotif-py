| Task | Size | Threads | R | Python | Speed-up | R memory | Python memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Read MEME motifs | 8,790 motifs | 1 | 1.4 s | 0.0339 s | 41x | 26 MB | 22 MB |
| Write MEME motifs | 8,790 motifs | 1 | 1.26 s | 0.0379 s | 33x | 100 MB | 20 MB |
| Convert motifs to PWMs | 87,900 motifs | 1 | 5.92 s | 0.115 s | 51x | 203 MB | 81 MB |
| Create a motif from sites | 2,000,000 sites | 1 | 1.67 s | 0.367 s | 5x | 1,071 MB | 306 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 1 | 36.1 s | 0.385 s | 94x | 20 MB | 10 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 2 | 18.2 s | 0.21 s | 87x | 24 MB | 74 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 1 | 35.9 s | 0.572 s | 63x | 25 MB | 47 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 2 | 18.1 s | 0.337 s | 54x | 29 MB | 110 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 1 | 24.5 s | 0.295 s | 83x | 138 MB | 26 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 2 | 12.5 s | 0.154 s | 81x | 146 MB | 24 MB |
| Enrichment of 100 motifs | 1,000 peaks | 1 | 35.6 s | 0.362 s | 98x | 82 MB | 36 MB |
| Enrichment of 100 motifs | 1,000 peaks | 2 | 18 s | 0.2 s | 90x | 83 MB | 36 MB |
| Score thresholds for 100 P-values | 879 motifs | 1 | 13.7 s | 3.5 s | 4x | 20 MB | 38 MB |
| Score thresholds for 100 P-values | 879 motifs | 2 | 6.94 s | 1.95 s | 4x | 28 MB | 46 MB |
| Compare all against all, with P-values | 879 motifs | 1 | 49.2 s | 2.1 s | 23x | 323 MB | 256 MB |
| Compare all against all, with P-values | 879 motifs | 2 | 25.2 s | 1.22 s | 21x | 317 MB | 257 MB |
| Merge similar motifs | 879 motifs | 1 | 48.8 s | 2.03 s | 24x | 330 MB | 237 MB |
| Merge similar motifs | 879 motifs | 2 | 25.3 s | 1.19 s | 21x | 318 MB | 240 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 1 | 3.57 s | 0.199 s | 18x | 206 MB | 118 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 2 | 2.08 s | 0.122 s | 17x | 215 MB | 119 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 1 | 11 s | 0.679 s | 16x | 219 MB | 118 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 2 | 6.08 s | 0.442 s | 14x | 223 MB | 120 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 1 | 15.2 s | 0.702 s | 22x | 806 MB | 104 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 2 | 12.3 s | 0.425 s | 29x | 805 MB | 168 MB |
| Sequence complexity | 100,000 500 bp sequences | 1 | 25 s | 0.769 s | 32x | 2,653 MB | 591 MB |
| Sequence complexity | 100,000 500 bp sequences | 2 | 22.4 s | 0.561 s | 40x | 2,782 MB | 654 MB |
