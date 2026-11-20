"""Whether the models respond to each motif: global importance analysis.

Each motif is planted in the middle of background sequences, and the model
predicts with and without it. The backgrounds are windows of the model's own
regions, widened to its input and shuffled by pairs of letters, which keeps
their composition but breaks up any motif. Each background gets its own
instance of the motif, drawn from its PPM, on a strand at random. Copies of
the motif with the columns shuffled are planted the same way, so that a
motif counts only for what its letters' order adds, not for their
composition: planting any run of GCs can open chromatin, say.

    python insertion.py prep    [--work ...] [--models ...]    # with universalmotif: motifs and backgrounds
    python insertion.py run     --model-kind tf|torch ...      # with TensorFlow 2.15 or PyTorch: predictions
    python insertion.py collect [--work ...]                   # with universalmotif: one table

The effect on a counts head is the change in predicted log counts, and on a
profile head the Jensen-Shannon distance between the profiles with and
without the motif. Both are given against the shuffled copies (effect),
along with a Wilcoxon signed-rank test over the backgrounds (pvalue). prep
writes work/insertion/<model>.<head>.npz, and the same for the calibration
motifs in work/insertion/random; run writes a .tsv beside each, and collect
work/checks/insertion.tsv and insertion_random.tsv.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INPUT = 2114
BACKGROUNDS = 128
CONTROLS = 2
FOLDS = 2
# Random motifs, made of columns drawn at random from all of JASPAR's motifs, planted on each head's
# backgrounds like the motifs themselves. Any two sequences differ a little in their effect on a model,
# so with 128 backgrounds even these pass the test of significance; their effects, which grow with a
# motif's width, set how large a motif's must be to count (checks.py). Half take their widths from the
# head's own motifs and half are spread evenly over MIN_WIDTH to MAX_WIDTH.
CALIBRATION = 200
MIN_WIDTH, MAX_WIDTH = 6, 50


# ---------------------------------------------------------------------------
# prep (with universalmotif)

def prep(args):
    import polars as pl
    import universalmotif as um

    sys.path.insert(0, str(HERE))
    import checks
    import harness

    work = pathlib.Path(args.work)
    out_dir = work / "insertion"
    out_dir.mkdir(parents=True, exist_ok=True)
    genome = um.Genome(work / "files" / "hg38.2bit")
    rows = harness.manifest(args.models)
    backgrounds = {}
    for row in rows.iter_rows(named=True):
        d = work / "results" / row["model"] / row["head"]
        if not d.is_dir():
            continue
        if row["model"] not in backgrounds:
            backgrounds[row["model"]] = _backgrounds(um, pl, work, row, genome, args.seed)
        name = f"{row['model']}.{row['head']}.npz"
        found = checks._motifs(um, work, row)
        if not found:
            continue
        _save(out_dir / name, backgrounds[row["model"]], found, np.random.default_rng(args.seed))
        widths = [_span(m.width, (m.gaploc[0], m.mingap[0], m.maxgap[0]) if m.isgapped else (0, 0, 0))
                  for ms in found.values() for m in ms]
        _save(out_dir / "random" / name, backgrounds[row["model"]],
              {"random": _random_motifs(um, work, widths, args.seed)}, np.random.default_rng(args.seed + 1))
        print(f"  {row['model']} {row['head']}: {sum(len(v) for v in found.values())} motifs", file=sys.stderr,
              flush=True)


def _save(path: pathlib.Path, backgrounds: np.ndarray, found: dict, rng):
    """The motifs of each method as PPMs, with CONTROLS copies of each with
    their columns shuffled, and the backgrounds to plant them in."""
    recs, mats, ctrl = [], [], []
    for method, ppms in found.items():
        for m in ppms:
            p = np.asarray(m.matrix, dtype=np.float64)
            p = (p + 1e-9) / (p + 1e-9).sum(0, keepdims=True)
            gap = (m.gaploc[0], m.mingap[0], m.maxgap[0]) if m.isgapped else (0, 0, 0)
            recs.append((method, m.name, m.consensus or "", *gap))
            mats.append(p.astype(np.float32))
            # Copies with the columns shuffled; a composite's halves are shuffled apart.
            cs = []
            for _ in range(CONTROLS):
                if m.isgapped:
                    k = m.gaploc[0]
                    cols = np.concatenate([rng.permutation(k), k + rng.permutation(p.shape[1] - k)])
                else:
                    cols = rng.permutation(p.shape[1])
                cs.append(p[:, cols].astype(np.float32))
            ctrl.append(cs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, backgrounds=backgrounds,
        method=np.array([r[0] for r in recs]), name=np.array([r[1] for r in recs]),
        consensus=np.array([r[2] for r in recs]), gap=np.array([r[3:] for r in recs], dtype=np.int64),
        widths=np.array([m.shape[1] for m in mats]),
        matrices=np.concatenate([m.T for m in mats]),
        controls=np.concatenate([c.T for cs in ctrl for c in cs]))


def _random_motifs(um, work: pathlib.Path, widths: list[int], seed: int) -> list:
    """CALIBRATION random motifs, each of columns drawn at random from all of
    JASPAR's: half with widths drawn from widths, the head's own motifs', and
    half spread evenly over MIN_WIDTH to MAX_WIDTH. Their columns are as
    informative as real ones, but no model can have learned their sites."""
    jaspar = um.read_meme(work / "files" / "jaspar.meme")
    pool = np.concatenate([np.asarray(m.matrix, dtype=float) / np.asarray(m.matrix, dtype=float).sum(0)
                           for m in jaspar], axis=1)
    rng = np.random.default_rng(seed)
    half = CALIBRATION // 2
    own = np.clip(rng.choice(widths, half), MIN_WIDTH, MAX_WIDTH) if widths else np.array([], dtype=int)
    even = np.linspace(MIN_WIDTH, MAX_WIDTH, CALIBRATION - len(own)).round().astype(int)
    return [um.create_motif(pool[:, rng.choice(pool.shape[1], int(w))], type="PPM", name=f"random_{k + 1}",
                            alphabet="DNA") for k, w in enumerate(np.concatenate([own, even]))]


def _span(width: int, gap) -> int:
    """The letters a planted motif covers, with a composite's gap at the
    middle of its range."""
    loc, lo, hi = gap
    return int(width + ((lo + hi) // 2 if loc else 0))


def _geometry(z) -> tuple[np.ndarray, np.ndarray]:
    """The span (as _span()) and total information content of each motif of a
    prep file, in its order."""
    offsets = np.concatenate([[0], np.cumsum(z["widths"])])
    span, ic = [], []
    for i in range(len(z["widths"])):
        p = z["matrices"][offsets[i]:offsets[i + 1]].astype(float)
        span.append(_span(int(z["widths"][i]), tuple(z["gap"][i])))
        ic.append(float((2 + (p * np.log2(p)).sum(1)).sum()))
    return np.array(span), np.array(ic)


def _backgrounds(um, pl, work, row, genome, seed):
    """BACKGROUNDS of the model's windows, widened to INPUT about their
    middles and shuffled by pairs of letters, as letter codes 0-3."""
    w = pl.read_parquet(work / "data" / row["model"] / row["head"] / "windows.parquet")
    rng = np.random.default_rng(seed)
    w = w[np.sort(rng.choice(w.height, BACKGROUNDS * 2, replace=False))]
    mid = (w["start"] + w["end"]) // 2
    ranges = pl.DataFrame({"chrom": w["chrom"], "start": mid - INPUT // 2, "end": mid - INPUT // 2 + INPUT})
    seqs = [s.upper() for s in um.extract_ranges(genome, ranges).values()]
    seqs = [s for s in seqs if len(s) == INPUT and s.count("N") == 0][:BACKGROUNDS]
    seqs = um.shuffle_sequences(seqs, k=2, seed=seed)
    lut = np.full(256, 0, dtype=np.uint8)
    for k, b in enumerate("ACGT"):
        lut[ord(b)] = k
    return np.stack([lut[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs])


# ---------------------------------------------------------------------------
# run (with TensorFlow or PyTorch)

class Predictor:
    """The model's folds, as one function from one-hot sequences (n, INPUT,
    4) to (log counts, profile probabilities), averaged over the folds."""

    def __init__(self, kind: str, model: str, head: str, models_dir: pathlib.Path):
        self.kind, self.head = kind, head
        d = models_dir / model
        self.folds = []
        if kind == "tf":
            import tensorflow as tf

            for g in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(g, True)
            self.tf = tf
            if model.startswith("chrombpnet"):
                from tensorflow.keras.utils import get_custom_objects

                get_custom_objects().update({"tf": tf})
                for f in sorted(glob.glob(str(d / "fold_*" / "model.chrombpnet_nobias.*.h5")))[:FOLDS]:
                    m = tf.keras.models.load_model(f, compile=False)
                    self.folds.append(tf.function(lambda x, m=m: m(x, training=False)))
                self.style = "chrombpnet"
            else:
                for f in sorted(glob.glob(str(d / "fold_*" / "saved" / "*")))[:FOLDS]:
                    self.folds.append(tf.saved_model.load(f).signatures["serving_default"])
                self.style = "bpnet"
        else:
            import torch

            self.torch = torch
            for f in sorted(glob.glob(str(d / "fold_*" / "*.state_dict.torch")))[:FOLDS]:
                m = _procapnet(torch)
                m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
                self.folds.append(m.cuda().eval())
            self.style = "procapnet"
        if not self.folds:
            raise FileNotFoundError(f"no models for {model} in {d}")

    def __call__(self, x: np.ndarray, batch: int = 512):
        counts, profiles = [], []
        for s in range(0, len(x), batch):
            c, p = self._one(x[s:s + batch])
            counts.append(c)
            profiles.append(p)
        return np.concatenate(counts), np.concatenate(profiles)

    def _one(self, x):
        cs, ps = [], []
        for f in self.folds:
            if self.style == "chrombpnet":
                prof, cnt = f(self.tf.constant(x))
                prof, cnt = prof.numpy(), cnt.numpy()[:, 0]
            elif self.style == "bpnet":
                n = len(x)
                out = f(sequence=self.tf.constant(x), profile_bias_input_0=self.tf.zeros((n, 1000, 2)),
                        counts_bias_input_0=self.tf.zeros((n, 2)))
                prof = out["profile_predictions"].numpy().reshape(n, -1)
                cnt = out["logcounts_predictions"].numpy()[:, 0]
            else:
                with self.torch.no_grad():
                    prof, cnt = f(self.torch.tensor(x.transpose(0, 2, 1)).cuda())
                prof, cnt = prof.reshape(len(x), -1).cpu().numpy(), cnt.cpu().numpy()[:, 0]
            prof = np.exp(prof - prof.max(1, keepdims=True))
            ps.append(prof / prof.sum(1, keepdims=True))
            cs.append(cnt)
        return np.mean(cs, 0), np.mean(ps, 0)


def _procapnet(torch):
    """ProCapNet as given in the README of its ENCODE models."""

    class ProCapNet(torch.nn.Module):
        def __init__(self, n_filters=512, n_layers=8, n_outputs=2):
            super().__init__()
            self.n_layers, self.trimming, self.k = n_layers, 557, 75
            self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
            self.rconvs = torch.nn.ModuleList([torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2 ** i,
                                                               dilation=2 ** i) for i in range(1, n_layers + 1)])
            self.fconv = torch.nn.Conv1d(n_filters, n_outputs, kernel_size=self.k)
            self.linear = torch.nn.Linear(n_filters, 1)

        def forward(self, X):
            start, end = self.trimming, X.shape[2] - self.trimming
            X = torch.relu(self.iconv(X))
            for i in range(self.n_layers):
                X = X + torch.relu(self.rconvs[i](X))
            X = X[:, :, start - self.k // 2:end + self.k // 2]
            return self.fconv(X), self.linear(torch.mean(X, axis=2)).reshape(X.shape[0], 1)

    return ProCapNet()


def _plant(bg: np.ndarray, p: np.ndarray, gap, rng) -> np.ndarray:
    """Copies of the backgrounds (codes) with an instance of the PPM p
    (width, 4) drawn for each and set in the middle, on a strand at random;
    a composite's gap takes a length in its range, keeping the background's
    letters."""
    n = len(bg)
    out = bg.copy()
    cum = np.cumsum(p, 1)
    draws = (rng.random((n, p.shape[0], 1)) > cum[None]).sum(2).clip(0, 3)
    flip = rng.random(n) < 0.5
    draws[flip] = 3 - draws[flip, ::-1]
    loc, lo, hi = gap
    for b in range(n):
        site = draws[b]
        if loc:
            g = rng.integers(lo, hi + 1)
            k = loc if not flip[b] else p.shape[0] - loc
            parts = [(0, site[:k]), (k + g, site[k:])]
            width = p.shape[0] + g
        else:
            parts, width = [(0, site)], p.shape[0]
        s0 = INPUT // 2 - width // 2
        for off, letters in parts:
            out[b, s0 + off:s0 + off + len(letters)] = letters
    return out


def _jsd(p, q):
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log(np.maximum(a, 1e-30) / np.maximum(b, 1e-30)), 0), 1)
    return np.sqrt(np.maximum((kl(p, m) + kl(q, m)) / 2, 0) / np.log(2))


def run(args):
    from scipy.stats import wilcoxon

    work = pathlib.Path(args.work)
    models_dir = pathlib.Path(args.models_dir) if args.models_dir else work / "models"
    files = sorted((work / "insertion").glob("*.npz")) + sorted((work / "insertion" / "random").glob("*.npz"))
    if args.models:
        import re

        files = [f for f in files if any(re.search(p, f.name.split(".")[0]) for p in args.models)]
    eye = np.eye(4, dtype=np.float32)
    for f in files:
        model, head = f.name[:-4].rsplit(".", 1)
        out = f.with_suffix(".tsv")
        if out.exists() and not args.force:
            continue
        kind = "torch" if model.startswith("procapnet") else "tf"
        if kind != args.model_kind:
            continue
        z = np.load(f)
        pred = Predictor(kind, model, head, models_dir)
        bg = z["backgrounds"]
        base_c, base_p = pred(eye[bg])
        widths = z["widths"]
        offsets = np.concatenate([[0], np.cumsum(widths)])
        coffsets = np.concatenate([[0], np.cumsum(np.repeat(widths, CONTROLS))])
        cache = {}
        lines = ["method\tmotif\tconsensus\teffect\traw\tcontrol\tpvalue"]
        rng = np.random.default_rng(1)
        for i in range(len(widths)):
            p = z["matrices"][offsets[i]:offsets[i + 1]]
            gap = tuple(int(v) for v in z["gap"][i])
            key = hashlib.sha1(np.round(p, 3).tobytes() + str(gap).encode()).hexdigest()
            if key not in cache:
                sets = [p] + [z["controls"][coffsets[i * CONTROLS + c]:coffsets[i * CONTROLS + c + 1]]
                              for c in range(CONTROLS)]
                planted = np.concatenate([_plant(bg, q, gap, rng) for q in sets])
                c, pr = pred(eye[planted])
                n = len(bg)
                if head == "counts":
                    d = c - np.tile(base_c, len(sets))
                else:
                    d = _jsd(np.tile(base_p, (len(sets), 1)), pr)
                d = d.reshape(len(sets), n)
                diff = d[0] - d[1:].mean(0)
                pval = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                cache[key] = (float(diff.mean()), float(d[0].mean()), float(d[1:].mean()), pval)
            e, raw, ctrl, pval = cache[key]
            lines.append(f"{z['method'][i]}\t{z['name'][i]}\t{z['consensus'][i]}\t{e:.5g}\t{raw:.5g}\t{ctrl:.5g}\t{pval:.4g}")
        out.write_text("\n".join(lines) + "\n")
        print(f"  {model} {head}: {len(widths)} motifs, {len(cache)} distinct", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# collect (with universalmotif)

def collect(args):
    import polars as pl

    work = pathlib.Path(args.work)
    (work / "checks").mkdir(exist_ok=True)
    for d, name in [(work / "insertion", "insertion.tsv"), (work / "insertion" / "random", "insertion_random.tsv")]:
        parts = []
        for f in sorted(d.glob("*.tsv")):
            model, head = f.name[:-4].rsplit(".", 1)
            part = pl.read_csv(f, separator="\t", infer_schema_length=None)
            # Rows are in the order of the prep file, which gives each motif's span and information content.
            span, ic = _geometry(np.load(f.with_suffix(".npz")))
            if len(span) != part.height:
                raise ValueError(f"{f} and its prep file disagree; rerun insertion.py run for it")
            parts.append(part.with_columns(model=pl.lit(model), head=pl.lit(head), width=pl.Series(span),
                                           ic=pl.Series(ic)))
        df = pl.concat(parts)
        df.write_csv(work / "checks" / name, separator="\t")
        print(f"{df.height} rows in {work / 'checks' / name}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep", "run", "collect"])
    ap.add_argument("--work", default=str(HERE / "work"))
    ap.add_argument("--models", nargs="*", help="regular expressions matched against model names")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model-kind", choices=["tf", "torch"], help="run: TensorFlow or PyTorch models")
    ap.add_argument("--models-dir", help="run: the trained models (default: models in --work)")
    ap.add_argument("--force", action="store_true", help="run: redo heads already done")
    args = ap.parse_args()
    {"prep": prep, "run": run, "collect": collect}[args.stage](args)


if __name__ == "__main__":
    main()
