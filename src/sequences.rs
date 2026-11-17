//! Bindings for k-mers, backgrounds, complexity, shuffling, random sequences,
//! implanting and motif discovery.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;
use universalmotif::Motif;
use universalmotif::complexity::{self, ComplexityMethod, ComplexityOptions};
use universalmotif::finder::{FinderOptions, motif_finder};
use universalmotif::gapped::{GappedFinderOptions, gapped_motif_finder};
use universalmotif::kmer::{self, BkgOptions, WindowOptions};
use universalmotif::sample::{self, ImplantMode, ImplantOptions};
use universalmotif::select::{
    RefineOptions, SelectOptions, Site, StabilityOptions, motif_stability, refine_motifs,
    select_motifs,
};
use universalmotif::shuffle::{self, ShuffleMethod, ShuffleOptions};

use pyo3_arrow::PyTable;

use crate::arrow::{f64_vec, f64s, i64s, strs, table};
use crate::{PyMotif, err};

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn window(w: Option<(f64, f64)>) -> Option<WindowOptions> {
    w.map(|(size, overlap)| WindowOptions { size, overlap })
}

#[pyfunction]
fn _kmers(letters: &str, k: usize) -> Vec<String> {
    kmer::kmers(&chars(letters), k)
}

#[pyfunction]
fn _count_kmers(seq: &str, k: usize, alphabet: &str) -> Vec<u64> {
    kmer::count_kmers(seq, k, &chars(alphabet))
}

#[pyfunction]
fn _windows<'py>(
    py: Python<'py>,
    len: usize,
    window: usize,
    overlap: usize,
    incomplete: bool,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let w = kmer::windows(len, window, overlap, incomplete);
    let start: Vec<i64> = w.iter().map(|&(s, _)| s as i64).collect();
    let end: Vec<i64> = w.iter().map(|&(_, e)| e as i64).collect();
    (start.into_pyarray(py), end.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (seqs, method, alph, max_word))]
fn _complexity(
    seqs: Vec<PyBackedStr>,
    method: &str,
    alph: Option<String>,
    max_word: usize,
) -> PyResult<Vec<f64>> {
    let m = method.parse::<ComplexityMethod>().map_err(err)?;
    Ok(seqs
        .iter()
        .map(|s| complexity::complexity(s, m, alph.as_deref(), max_word))
        .collect())
}

#[pyfunction]
#[pyo3(signature = (seqs, names, method, window, overlap, alphabet, max_word, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _sequence_complexity(
    py: Python<'_>,
    seqs: Vec<PyBackedStr>,
    names: Vec<String>,
    method: &str,
    window: f64,
    overlap: Option<f64>,
    alphabet: Option<String>,
    max_word: usize,
    nthreads: usize,
) -> PyResult<PyTable> {
    let opts = ComplexityOptions {
        method: method.parse::<ComplexityMethod>().map_err(err)?,
        window,
        overlap,
        alphabet,
        max_word,
        nthreads,
    };
    let rows = py
        .detach(|| complexity::sequence_complexity(&seqs, &opts))
        .map_err(err)?;
    table(vec![
        (
            "sequence",
            strs(rows.iter().map(|r| names[r.sequence].as_str())),
        ),
        ("start", i64s(rows.iter().map(|r| r.start as i64))),
        ("end", i64s(rows.iter().map(|r| r.end as i64))),
        ("complexity", f64s(rows.iter().map(|r| r.complexity))),
    ])
}

/// k-mer counts as a table: `sequence` for per-sequence counts (reverse
/// complements take their sequence's name) and `start`/`end` for windows.
#[pyfunction]
#[pyo3(signature = (seqs, names, k, pseudocount, alphabet, rc, merge, window, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _get_bkg(
    py: Python<'_>,
    seqs: Vec<PyBackedStr>,
    names: Vec<String>,
    k: Vec<usize>,
    pseudocount: u64,
    alphabet: Option<String>,
    rc: bool,
    merge: bool,
    window: Option<(f64, f64)>,
    nthreads: usize,
) -> PyResult<PyTable> {
    let opts = BkgOptions {
        k,
        pseudocount,
        alphabet: alphabet.as_deref().map(chars),
        rc,
        merge,
        window: self::window(window),
        nthreads,
    };
    let t = py
        .detach(|| kmer::get_bkg_table(&seqs, &names, &opts))
        .map_err(err)?;
    // Each column is freed once it has been copied.
    let kmer::BkgTable {
        kmers,
        sequence,
        start,
        end,
        kmer,
        count,
        probability,
        ..
    } = t;
    let mut cols = Vec::new();
    if !merge {
        let n = names.len().max(1);
        cols.push((
            "sequence",
            strs(sequence.iter().map(|&s| names[s as usize % n].as_str())),
        ));
    }
    drop(sequence);
    if opts.window.is_some() {
        cols.push(("start", i64s(start.iter().map(|&s| s as i64))));
        cols.push(("end", i64s(end.iter().map(|&e| e as i64))));
    }
    drop((start, end));
    cols.push((
        "kmer",
        strs(kmer.iter().map(|&k| kmers[k as usize].as_str())),
    ));
    drop(kmer);
    cols.push(("count", i64s(count.iter().map(|&c| c as i64))));
    drop(count);
    cols.push(("probability", f64_vec(probability)));
    table(cols)
}

#[pyfunction]
#[pyo3(signature = (seqs, k, method, seed, window, nthreads))]
fn _shuffle_sequences(
    py: Python<'_>,
    seqs: Vec<PyBackedStr>,
    k: usize,
    method: &str,
    seed: u64,
    window: Option<(f64, f64)>,
    nthreads: usize,
) -> PyResult<Vec<String>> {
    let opts = ShuffleOptions {
        k,
        method: method.parse::<ShuffleMethod>().map_err(err)?,
        seed,
        window: self::window(window),
        nthreads,
    };
    py.detach(|| shuffle::shuffle_sequences(&seqs, &opts))
        .map_err(err)
}

/// `n` shuffles of each row of `codes`, letters 0 to 3 with 4 for N kept in
/// place, as an array of shape (sequences, n, length).
#[pyfunction]
#[pyo3(signature = (codes, n, k, method, seed, offset, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _shuffle_codes<'py>(
    py: Python<'py>,
    codes: numpy::PyReadonlyArray2<'py, u8>,
    n: usize,
    k: usize,
    method: &str,
    seed: u64,
    offset: u64,
    nthreads: usize,
) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};

    let (rows, length) = (codes.shape()[0], codes.shape()[1]);
    let values = codes
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let opts = shuffle::CodeShuffleOptions {
        k,
        method: method.parse::<ShuffleMethod>().map_err(err)?,
        seed,
        offset,
        unknown: Some(4),
        nthreads,
    };
    let out = py
        .detach(|| shuffle::shuffle_codes(values, length, n, &opts))
        .map_err(err)?;
    out.into_pyarray(py).reshape([rows, n, length])
}

#[pyfunction]
#[pyo3(signature = (alphabet, n, len, freqs, seed, nthreads))]
fn _create_sequences(
    py: Python<'_>,
    alphabet: &str,
    n: usize,
    len: usize,
    freqs: Option<Vec<f64>>,
    seed: u64,
    nthreads: usize,
) -> PyResult<Vec<String>> {
    let a = chars(alphabet);
    py.detach(|| shuffle::create_sequences(&a, n, len, freqs.as_deref(), seed, nthreads))
        .map_err(err)
}

#[pyfunction]
fn _shuffle_motifs(motifs: Vec<PyRef<'_, PyMotif>>, k: usize, seed: u64) -> PyResult<Vec<PyMotif>> {
    let motifs: Vec<Motif> = motifs.iter().map(|m| m.inner.clone()).collect();
    shuffle::shuffle_motifs(&motifs, k, seed)
        .map(|v| v.into_iter().map(PyMotif::from).collect())
        .map_err(err)
}

#[pyfunction]
fn _sample_sites(motif: PyRef<'_, PyMotif>, n: usize, seed: u64) -> PyResult<Vec<String>> {
    sample::sample_sites(&motif.inner, n, seed).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (motifs, seqs, n_per_seq, rate, positions, both_strands, centre_bias, min_spacing, max_retries, seed))]
#[allow(clippy::too_many_arguments)]
fn _implant_motifs<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    seqs: Vec<PyBackedStr>,
    n_per_seq: usize,
    rate: Option<f64>,
    positions: Option<Vec<Vec<usize>>>,
    both_strands: bool,
    centre_bias: usize,
    min_spacing: usize,
    max_retries: usize,
    seed: u64,
) -> PyResult<(Vec<String>, Bound<'py, PyDict>, Vec<String>)> {
    let motifs: Vec<Motif> = motifs.iter().map(|m| m.inner.clone()).collect();
    let mode = match (positions, rate) {
        (Some(p), _) => ImplantMode::Positions(p),
        (None, Some(r)) => ImplantMode::Rate(r),
        (None, None) => ImplantMode::Count(n_per_seq),
    };
    let opts = ImplantOptions {
        mode,
        both_strands,
        centre_bias,
        min_spacing,
        max_retries,
        seed,
    };
    let r = py
        .detach(|| sample::implant_motifs(&motifs, &seqs, &opts))
        .map_err(err)?;
    let d = PyDict::new(py);
    let im = &r.implants;
    d.set_item(
        "sequence_i",
        im.iter().map(|x| x.sequence).collect::<Vec<_>>(),
    )?;
    d.set_item("motif_i", im.iter().map(|x| x.motif).collect::<Vec<_>>())?;
    d.set_item("start", im.iter().map(|x| x.start).collect::<Vec<_>>())?;
    d.set_item("width", im.iter().map(|x| x.width).collect::<Vec<_>>())?;
    d.set_item(
        "strand",
        im.iter()
            .map(|x| if x.minus_strand { "-" } else { "+" })
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "planted",
        im.iter().map(|x| x.planted.as_str()).collect::<Vec<_>>(),
    )?;
    Ok((r.sequences, d, r.warnings))
}

#[pyfunction]
#[pyo3(signature = (seqs, negatives, k, min_gap, max_gap, nmotifs, hit_pvalue, stop_pvalue, flank, min_ic, n_seed_pairs, min_fraction, holdout, mismatch, mask, rc, shuffle_k, bkg_shuffles, seed, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _gapped_motif_finder<'py>(
    py: Python<'py>,
    seqs: Vec<PyBackedStr>,
    negatives: Option<Vec<PyBackedStr>>,
    k: Vec<usize>,
    min_gap: usize,
    max_gap: usize,
    nmotifs: usize,
    hit_pvalue: f64,
    stop_pvalue: f64,
    flank: usize,
    min_ic: f64,
    n_seed_pairs: usize,
    min_fraction: f64,
    holdout: f64,
    mismatch: bool,
    mask: bool,
    rc: bool,
    shuffle_k: usize,
    bkg_shuffles: usize,
    seed: u64,
    nthreads: usize,
) -> PyResult<(Vec<PyMotif>, Bound<'py, PyDict>)> {
    let opts = GappedFinderOptions {
        k,
        min_gap,
        max_gap,
        nmotifs,
        hit_pvalue,
        stop_pvalue,
        flank,
        min_ic,
        n_seed_pairs,
        min_fraction,
        holdout,
        mismatch,
        mask,
        rc,
        shuffle_k,
        bkg_shuffles,
        seed,
        nthreads,
    };
    let found = py
        .detach(|| gapped_motif_finder(&seqs, negatives.as_deref(), &opts))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("k", found.iter().map(|f| f.k).collect::<Vec<_>>())?;
    d.set_item(
        "seed_pair",
        found
            .iter()
            .map(|f| f.seed_pair.clone())
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "seqs_pos",
        found.iter().map(|f| f.seqs_pos).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "seqs_neg",
        found.iter().map(|f| f.seqs_neg).collect::<Vec<_>>(),
    )?;
    d.set_item("n_pos", found.iter().map(|f| f.n_pos).collect::<Vec<_>>())?;
    d.set_item("n_neg", found.iter().map(|f| f.n_neg).collect::<Vec<_>>())?;
    d.set_item(
        "candidates",
        found.iter().map(|f| f.candidates).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "seed_pairs",
        found.iter().map(|f| f.seed_pairs).collect::<Vec<_>>(),
    )?;
    d.set_item("pvalue", found.iter().map(|f| f.pvalue).collect::<Vec<_>>())?;
    d.set_item(
        "log10_pvalue",
        found.iter().map(|f| f.log10_pvalue).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "spacing_pvalue",
        found.iter().map(|f| f.spacing_pvalue).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "log10_spacing_pvalue",
        found
            .iter()
            .map(|f| f.log10_spacing_pvalue)
            .collect::<Vec<_>>(),
    )?;
    Ok((
        found.into_iter().map(|f| PyMotif::from(f.motif)).collect(),
        d,
    ))
}

#[pyfunction]
#[pyo3(signature = (seqs, negatives, min_width, max_width, nmotifs, hit_pvalue, stop_pvalue, qvalue, dedup_overlap, rc, shuffle_k, bkg_shuffles, seed, pseudocount, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _motif_finder<'py>(
    py: Python<'py>,
    seqs: Vec<PyBackedStr>,
    negatives: Option<Vec<PyBackedStr>>,
    min_width: usize,
    max_width: usize,
    nmotifs: usize,
    hit_pvalue: f64,
    stop_pvalue: f64,
    qvalue: f64,
    dedup_overlap: f64,
    rc: bool,
    shuffle_k: usize,
    bkg_shuffles: usize,
    seed: u64,
    pseudocount: u32,
    nthreads: usize,
) -> PyResult<(Vec<PyMotif>, Bound<'py, PyDict>)> {
    let opts = FinderOptions {
        min_width,
        max_width,
        nmotifs,
        hit_pvalue,
        stop_pvalue,
        qvalue,
        dedup_overlap,
        rc,
        shuffle_k,
        bkg_shuffles,
        seed,
        pseudocount,
        nthreads,
    };
    let found = py
        .detach(|| motif_finder(&seqs, negatives.as_deref(), &opts))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("rank", found.iter().map(|f| f.rank).collect::<Vec<_>>())?;
    d.set_item(
        "width",
        found.iter().map(|f| f.motif.width()).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "seqs_pos",
        found.iter().map(|f| f.seqs_pos).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "seqs_neg",
        found.iter().map(|f| f.seqs_neg).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "sites_pos",
        found.iter().map(|f| f.sites_pos).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "sites_neg",
        found.iter().map(|f| f.sites_neg).collect::<Vec<_>>(),
    )?;
    d.set_item("n_pos", found.iter().map(|f| f.n_pos).collect::<Vec<_>>())?;
    d.set_item("n_neg", found.iter().map(|f| f.n_neg).collect::<Vec<_>>())?;
    d.set_item(
        "candidates",
        found.iter().map(|f| f.candidates).collect::<Vec<_>>(),
    )?;
    d.set_item("pvalue", found.iter().map(|f| f.pvalue).collect::<Vec<_>>())?;
    d.set_item(
        "log10_pvalue",
        found.iter().map(|f| f.log10_pvalue).collect::<Vec<_>>(),
    )?;
    d.set_item("qvalue", found.iter().map(|f| f.qvalue).collect::<Vec<_>>())?;
    d.set_item(
        "log10_qvalue",
        found.iter().map(|f| f.log10_qvalue).collect::<Vec<_>>(),
    )?;
    Ok((
        found.into_iter().map(|f| PyMotif::from(f.motif)).collect(),
        d,
    ))
}

fn sites_from(sites: Option<Vec<(usize, usize, usize)>>) -> Option<Vec<Site>> {
    sites.map(|v| {
        v.into_iter()
            .map(|(sequence, start, end)| Site {
                sequence,
                start,
                end,
            })
            .collect()
    })
}

#[pyfunction]
#[pyo3(signature = (motifs, sequences, sites, similarity, pvalue, min_gain, min_overlap, max_similarity, rc, nthreads, reachable=false))]
#[allow(clippy::too_many_arguments)]
fn _select_motifs<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    sequences: Vec<PyBackedStr>,
    sites: Option<Vec<(usize, usize, usize)>>,
    similarity: Option<Vec<Vec<f64>>>,
    pvalue: f64,
    min_gain: f64,
    min_overlap: usize,
    max_similarity: Option<f64>,
    rc: bool,
    nthreads: usize,
    reachable: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let motifs: Vec<Motif> = motifs.iter().map(|m| m.inner.clone()).collect();
    let sites = sites_from(sites);
    let opts = SelectOptions {
        pvalue,
        min_gain,
        min_overlap,
        max_similarity,
        rc,
        nthreads,
        reachable,
    };
    let sel = py
        .detach(|| {
            select_motifs(
                &motifs,
                &sequences,
                sites.as_deref(),
                similarity.as_deref(),
                &opts,
            )
        })
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("n_targets", sel.n_targets)?;
    d.set_item("targets", sel.targets)?;
    d.set_item("rank", sel.rank)?;
    d.set_item("gain", sel.gain)?;
    d.set_item("cumulative", sel.cumulative)?;
    d.set_item("kept", sel.kept)?;
    d.set_item("similar_to", sel.similar_to)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (motifs, sequences, sites, pvalue, flank, iterations, competitive, min_ic, max_width, min_sites, min_overlap, rc, nthreads, reachable=false))]
#[allow(clippy::too_many_arguments)]
fn _refine_motifs(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    sequences: Vec<PyBackedStr>,
    sites: Option<Vec<(usize, usize, usize)>>,
    pvalue: f64,
    flank: usize,
    iterations: usize,
    competitive: bool,
    min_ic: f64,
    max_width: usize,
    min_sites: usize,
    min_overlap: usize,
    rc: bool,
    nthreads: usize,
    reachable: bool,
) -> PyResult<(Vec<PyMotif>, Vec<usize>)> {
    let motifs: Vec<Motif> = motifs.iter().map(|m| m.inner.clone()).collect();
    let sites = sites_from(sites);
    let opts = RefineOptions {
        pvalue,
        flank,
        iterations,
        competitive,
        min_ic,
        max_width,
        min_sites,
        min_overlap,
        rc,
        nthreads,
        reachable,
    };
    let out = py
        .detach(|| refine_motifs(&motifs, &sequences, sites.as_deref(), &opts))
        .map_err(err)?;
    let used = out.iter().map(|r| r.sites).collect();
    Ok((
        out.into_iter().map(|r| PyMotif::from(r.motif)).collect(),
        used,
    ))
}

#[pyfunction]
#[pyo3(signature = (motifs, seqs, negatives, min_width, max_width, nmotifs, hit_pvalue, stop_pvalue, qvalue, dedup_overlap, rc, shuffle_k, bkg_shuffles, seed, pseudocount, nthreads, rounds, fraction, min_similarity))]
#[allow(clippy::too_many_arguments)]
fn _motif_stability(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    seqs: Vec<PyBackedStr>,
    negatives: Option<Vec<PyBackedStr>>,
    min_width: usize,
    max_width: usize,
    nmotifs: usize,
    hit_pvalue: f64,
    stop_pvalue: f64,
    qvalue: f64,
    dedup_overlap: f64,
    rc: bool,
    shuffle_k: usize,
    bkg_shuffles: usize,
    seed: u64,
    pseudocount: u32,
    nthreads: usize,
    rounds: usize,
    fraction: f64,
    min_similarity: f64,
) -> PyResult<Vec<f64>> {
    let motifs: Vec<Motif> = motifs.iter().map(|m| m.inner.clone()).collect();
    let finder = FinderOptions {
        min_width,
        max_width,
        nmotifs,
        hit_pvalue,
        stop_pvalue,
        qvalue,
        dedup_overlap,
        rc,
        shuffle_k,
        bkg_shuffles,
        seed,
        pseudocount,
        nthreads,
    };
    let opts = StabilityOptions {
        rounds,
        fraction,
        min_similarity,
        seed,
    };
    py.detach(|| motif_stability(&motifs, &seqs, negatives.as_deref(), &finder, &opts))
        .map_err(err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(_count_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(_windows, m)?)?;
    m.add_function(wrap_pyfunction!(_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(_sequence_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(_get_bkg, m)?)?;
    m.add_function(wrap_pyfunction!(_shuffle_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(_shuffle_codes, m)?)?;
    m.add_function(wrap_pyfunction!(_create_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(_shuffle_motifs, m)?)?;
    m.add_function(wrap_pyfunction!(_sample_sites, m)?)?;
    m.add_function(wrap_pyfunction!(_implant_motifs, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_finder, m)?)?;
    m.add_function(wrap_pyfunction!(_select_motifs, m)?)?;
    m.add_function(wrap_pyfunction!(_refine_motifs, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_stability, m)?)?;
    m.add_function(wrap_pyfunction!(_gapped_motif_finder, m)?)?;
    Ok(())
}
