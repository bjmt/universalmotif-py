"""One call of contribution_motifs() on prepared inputs, with 2 threads.

    python run_cm.py INDIR [hypothetical] [key=value ...]

Prints the result and the time of the call; run it under /usr/bin/time -l
for the time and peak memory of the whole process.
"""

import pathlib
import sys
import time

import numpy as np
import universalmotif as um

d = pathlib.Path(sys.argv[1])
args = sys.argv[2:]
hyp = np.load(d / "hyp.npy") if "hypothetical" in args else None
kw = {k: (None if v == "None" else int(v)) for k, v in (a.split("=") for a in args if "=" in a)}
seqs = ["".join(x) for x in np.array(list("ACGTN"))[np.load(d / "codes.npy")]]
observed = np.load(d / "observed.npy")
t0 = time.time()
found = um.contribution_motifs(observed, seqs, hypothetical=hyp, seed=1, nthreads=2, **kw)
print(f"{found!r}: call {time.time() - t0:.1f} s", flush=True)
