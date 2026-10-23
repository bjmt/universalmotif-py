//! Bindings for P-values, scanning and hit deduplication.

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyDict, PyList};
use universalmotif::Motif;
use universalmotif::Strand;
use universalmotif::cluster::{self, Linkage, Tree, TreeOptions};
use universalmotif::compare::{self, CompareOptions, NullModel};
use universalmotif::dedup::{Interval, dedup};
use universalmotif::merge::{self, MergeOptions, MergeSimilarOptions};
use universalmotif::packed::EncodedSequences;
use universalmotif::pvalue::{self, PvalueOptions};
use universalmotif::scan::{
    HitTable, MatchValue, NoOverlaps, OverlapPriority, ScanOptions, motif_matches, scan_encoded,
    scan_fasta, scan_table,
};

use pyo3_arrow::PyTable;

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::builder::StringViewBuilder;

use crate::arrow::{f64_vec, f64s, i64_vec, i64s, names_at, strs, table};
use crate::packed::PyPacked;
use crate::{PyMotif, err, parse_bkg};

fn unwrap(motifs: &[PyRef<'_, PyMotif>]) -> Vec<Motif> {
    motifs.iter().map(|m| m.inner.clone()).collect()
}

/// Backgrounds: `None`, one (a sequence of numbers or a mapping) for every
/// motif, or a list with one per motif.
fn backgrounds(motifs: &[Motif], bkg: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<f64>>> {
    let Some(b) = bkg.filter(|b| !b.is_none()) else {
        return Ok(Vec::new());
    };
    let per_motif = b
        .cast::<PyList>()
        .ok()
        .and_then(|l| l.get_item(0).ok())
        .is_some_and(|first| first.extract::<f64>().is_err());
    if per_motif {
        let list = b.cast::<PyList>()?;
        if list.len() != motifs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "give one background for all motifs or one per motif",
            ));
        }
        list.iter()
            .zip(motifs)
            .map(|(x, m)| parse_bkg(&x, &m.alphabet))
            .collect()
    } else {
        motifs.iter().map(|m| parse_bkg(b, &m.alphabet)).collect()
    }
}

#[pyfunction]
#[pyo3(signature = (motifs, queries, bkg, nthreads, to_score))]
fn _motif_pvalue(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    queries: Vec<(usize, f64)>,
    bkg: Option<&Bound<'_, PyAny>>,
    nthreads: usize,
    to_score: bool,
) -> PyResult<Vec<f64>> {
    let motifs = unwrap(&motifs);
    let opts = PvalueOptions {
        bkg: backgrounds(&motifs, bkg)?,
        nthreads,
    };
    py.detach(|| {
        if to_score {
            pvalue::pvalue_to_score(&motifs, &queries, &opts)
        } else {
            pvalue::motif_pvalue(&motifs, &queries, &opts)
        }
    })
    .map_err(err)
}

#[pyfunction]
#[pyo3(signature = (motifs, nthreads=1))]
fn _min_pvalues(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    nthreads: usize,
) -> PyResult<Vec<f64>> {
    let motifs = unwrap(&motifs);
    let opts = PvalueOptions {
        bkg: Vec::new(),
        nthreads,
    };
    py.detach(|| pvalue::min_pvalues(&motifs, &opts))
        .map_err(err)
}

fn scan_options(
    pvalue: f64,
    rc: bool,
    no_overlaps: Option<(String, bool, bool)>,
    nthreads: usize,
    use_gaps: bool,
    correct_gaps: bool,
) -> PyResult<ScanOptions> {
    let no_overlaps = no_overlaps
        .map(|(by, ignore_strand, ignore_motif)| {
            let by = match by.as_str() {
                "pvalue" => OverlapPriority::Pvalue,
                "score" => OverlapPriority::Score,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "no_overlaps_by must be 'pvalue' or 'score'",
                    ));
                }
            };
            Ok(NoOverlaps {
                by,
                ignore_strand,
                ignore_motif,
            })
        })
        .transpose()?;
    Ok(ScanOptions {
        pvalue,
        rc,
        no_overlaps,
        nthreads,
        use_gaps,
        correct_gaps,
    })
}

/// The matched letters of `n` hits as an Arrow column, with
/// `matched(i, buf)` writing hit `i`'s letters.
fn matched_column(n: usize, mut matched: impl FnMut(usize, &mut Vec<u8>)) -> ArrayRef {
    let mut letters = StringViewBuilder::with_capacity(n);
    let mut buf = Vec::new();
    for i in 0..n {
        matched(i, &mut buf);
        letters.append_value(String::from_utf8_lossy(&buf));
    }
    Arc::new(letters.finish())
}

/// The hit table as Arrow.
fn hit_table(
    t: HitTable,
    letters: ArrayRef,
    motif_names: &[String],
    sequence_names: &[String],
) -> PyResult<PyTable> {
    let strand = strs(t.minus.iter().map(|&m| if m { "-" } else { "+" }));
    let (motif, sequence) = (
        names_at(&t.motif, motif_names),
        names_at(&t.sequence, sequence_names),
    );
    table(vec![
        ("motif", motif),
        ("motif_i", i64_vec(t.motif)),
        ("sequence", sequence),
        ("sequence_i", i64_vec(t.sequence)),
        ("start", i64_vec(t.start)),
        ("end", i64_vec(t.end)),
        ("strand", strand),
        ("score", f64_vec(t.score)),
        ("score_pct", f64_vec(t.score_pct)),
        ("match", letters),
        ("pvalue", f64_vec(t.pvalue)),
    ])
}

#[pyfunction]
#[pyo3(signature = (motifs, sequences, pvalue, rc, no_overlaps, nthreads, use_gaps, correct_gaps, motif_names, sequence_names))]
#[allow(clippy::too_many_arguments)]
fn _scan_sequences<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    sequences: Vec<PyBackedStr>,
    pvalue: f64,
    rc: bool,
    no_overlaps: Option<(String, bool, bool)>,
    nthreads: usize,
    use_gaps: bool,
    correct_gaps: bool,
    motif_names: Vec<String>,
    sequence_names: Vec<String>,
) -> PyResult<PyTable> {
    let motifs = unwrap(&motifs);
    let opts = scan_options(pvalue, rc, no_overlaps, nthreads, use_gaps, correct_gaps)?;
    let t = py
        .detach(|| scan_table(&motifs, &sequences, &opts))
        .map_err(err)?;
    let letters = matched_column(t.len(), |i, buf| t.matched_into(i, &sequences, buf));
    hit_table(t, letters, &motif_names, &sequence_names)
}

/// A sparse sequences by motifs matrix of matches, counts or best scores, as
/// (rows, columns, sequence indices, motif indices, values).
type Matches<'py> = (
    usize,
    usize,
    Bound<'py, numpy::PyArray1<u32>>,
    Bound<'py, numpy::PyArray1<u32>>,
    Bound<'py, numpy::PyArray1<f64>>,
);

#[pyfunction]
#[pyo3(signature = (motifs, sequences, packed, pvalue, rc, no_overlaps, nthreads, use_gaps, value))]
#[allow(clippy::too_many_arguments)]
fn _motif_matches<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    sequences: Option<Vec<PyBackedStr>>,
    packed: Option<Bound<'py, PyPacked>>,
    pvalue: f64,
    rc: bool,
    no_overlaps: Option<(String, bool, bool)>,
    nthreads: usize,
    use_gaps: bool,
    value: &str,
) -> PyResult<Matches<'py>> {
    use numpy::IntoPyArray;

    let motifs = unwrap(&motifs);
    let opts = scan_options(pvalue, rc, no_overlaps, nthreads, use_gaps, false)?;
    let value = value.parse::<MatchValue>().map_err(err)?;
    let m = match (&packed, sequences) {
        (Some(p), _) => {
            let p = p.get();
            py.detach(|| motif_matches(&motifs, &p.seqs, &opts, value))
        }
        (None, Some(seqs)) => py.detach(|| {
            let alphabet = motifs
                .first()
                .map(|m| m.alphabet.clone())
                .ok_or_else(|| universalmotif::Error::Argument("no motifs to scan with".into()))?;
            let enc = EncodedSequences::new(&seqs, &alphabet, nthreads)?;
            motif_matches(&motifs, &enc, &opts, value)
        }),
        (None, None) => return Err(pyo3::exceptions::PyValueError::new_err("no sequences")),
    }
    .map_err(err)?;
    Ok((
        m.n_sequences,
        m.n_motifs,
        m.sequence.into_pyarray(py),
        m.motif.into_pyarray(py),
        m.value.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (motifs, packed, pvalue, rc, no_overlaps, nthreads, use_gaps, correct_gaps, motif_names))]
#[allow(clippy::too_many_arguments)]
fn _scan_packed<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    packed: Bound<'py, PyPacked>,
    pvalue: f64,
    rc: bool,
    no_overlaps: Option<(String, bool, bool)>,
    nthreads: usize,
    use_gaps: bool,
    correct_gaps: bool,
    motif_names: Vec<String>,
) -> PyResult<PyTable> {
    let motifs = unwrap(&motifs);
    let opts = scan_options(pvalue, rc, no_overlaps, nthreads, use_gaps, correct_gaps)?;
    let p = packed.get();
    let t = py
        .detach(|| scan_encoded(&motifs, &p.seqs, &opts))
        .map_err(err)?;
    let letters = matched_column(t.len(), |i, buf| t.matched_encoded_into(i, &p.seqs, buf));
    hit_table(t, letters, &motif_names, &p.names)
}

#[pyfunction]
#[pyo3(signature = (motifs, path, pvalue, rc, no_overlaps, nthreads, use_gaps, correct_gaps, motif_names))]
#[allow(clippy::too_many_arguments)]
fn _scan_fasta<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    path: std::path::PathBuf,
    pvalue: f64,
    rc: bool,
    no_overlaps: Option<(String, bool, bool)>,
    nthreads: usize,
    use_gaps: bool,
    correct_gaps: bool,
    motif_names: Vec<String>,
) -> PyResult<PyTable> {
    let motifs = unwrap(&motifs);
    let opts = scan_options(pvalue, rc, no_overlaps, nthreads, use_gaps, correct_gaps)?;
    let mut f = py
        .detach(|| scan_fasta(&motifs, &path, &opts))
        .map_err(err)?;
    let letters = matched_column(f.len(), |i, buf| {
        buf.clear();
        buf.extend_from_slice(f.matched(i).as_bytes());
    });
    let t = std::mem::take(&mut f.table);
    hit_table(t, letters, &motif_names, &f.names)
}

#[pyfunction]
#[pyo3(signature = (sequence, motif, strand, start, end, priority, ignore_strand, ignore_motif))]
#[allow(clippy::too_many_arguments)]
fn _dedup(
    sequence: Vec<usize>,
    motif: Vec<usize>,
    strand: Vec<i32>,
    start: Vec<i64>,
    end: Vec<i64>,
    priority: Vec<f64>,
    ignore_strand: bool,
    ignore_motif: bool,
) -> Vec<bool> {
    let hits: Vec<Interval> = (0..sequence.len())
        .map(|i| Interval {
            sequence: sequence[i],
            motif: motif[i],
            strand: match strand[i] {
                0 => Strand::Plus,
                1 => Strand::Minus,
                _ => Strand::Both,
            },
            start: start[i],
            end: end[i],
            priority: priority[i],
        })
        .collect();
    dedup(&hits, ignore_strand, ignore_motif)
}

fn null_model(null: &str) -> PyResult<NullModel> {
    match null {
        "empirical" => Ok(NullModel::Empirical),
        "parametric" => Ok(NullModel::Parametric),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "null must be 'empirical' or 'parametric'",
        )),
    }
}

fn bkg4(bkg: Option<Vec<f64>>) -> PyResult<[f64; 4]> {
    match bkg {
        None => Ok([0.25; 4]),
        Some(b) if b.len() == 4 => {
            if b.iter().any(|v| v.is_nan() || *v < 0.0)
                || (b.iter().sum::<f64>() - 1.0).abs() > 1e-6
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "bkg must be four non-negative values summing to 1",
                ));
            }
            Ok([b[0], b[1], b[2], b[3]])
        }
        Some(_) => Err(pyo3::exceptions::PyValueError::new_err(
            "bkg must have four values (A, C, G, T)",
        )),
    }
}

fn compare_options(
    min_overlap: usize,
    rc: bool,
    null: &str,
    bkg: Option<Vec<f64>>,
    nthreads: usize,
    complete: bool,
) -> PyResult<CompareOptions> {
    Ok(CompareOptions {
        min_overlap,
        rc,
        null: null_model(null)?,
        bkg: bkg4(bkg)?,
        complete,
        nthreads,
    })
}

#[pyfunction]
#[pyo3(signature = (motifs, min_overlap, rc, null, bkg, need_pvalues, nthreads, complete=false))]
#[allow(clippy::too_many_arguments)]
fn _compare_matrix<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    min_overlap: usize,
    rc: bool,
    null: &str,
    bkg: Option<Vec<f64>>,
    need_pvalues: bool,
    nthreads: usize,
    complete: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let motifs = unwrap(&motifs);
    let opts = compare_options(min_overlap, rc, null, bkg, nthreads, complete)?;
    let m = py
        .detach(|| compare::compare_matrix(&motifs, &opts, need_pvalues))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("score", m.scores)?;
    d.set_item("pvalue", m.pvalues)?;
    d.set_item("qvalue", m.qvalues)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (motifs, queries, min_overlap, rc, null, bkg, nthreads, names, complete=false))]
#[allow(clippy::too_many_arguments)]
fn _compare_table(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    queries: Vec<usize>,
    min_overlap: usize,
    rc: bool,
    null: &str,
    bkg: Option<Vec<f64>>,
    nthreads: usize,
    names: Vec<String>,
    complete: bool,
) -> PyResult<PyTable> {
    let motifs = unwrap(&motifs);
    let opts = compare_options(min_overlap, rc, null, bkg, nthreads, complete)?;
    let rows = py
        .detach(|| compare::compare_motifs(&motifs, &queries, &opts))
        .map_err(err)?;
    table(vec![
        (
            "subject",
            strs(rows.iter().map(|c| names[c.query].as_str())),
        ),
        ("subject_i", i64s(rows.iter().map(|c| c.query as i64))),
        (
            "target",
            strs(rows.iter().map(|c| names[c.target].as_str())),
        ),
        ("target_i", i64s(rows.iter().map(|c| c.target as i64))),
        ("offset", i64s(rows.iter().map(|c| c.offset))),
        ("strand", strs(rows.iter().map(|c| c.strand.as_str()))),
        ("overlap", i64s(rows.iter().map(|c| c.overlap as i64))),
        ("score", f64s(rows.iter().map(|c| c.score))),
        (
            "subject_consensus",
            strs(rows.iter().map(|c| c.query_consensus.as_str())),
        ),
        (
            "target_consensus",
            strs(rows.iter().map(|c| c.target_consensus.as_str())),
        ),
        ("pvalue", f64s(rows.iter().map(|c| c.pvalue))),
        ("qvalue", f64s(rows.iter().map(|c| c.qvalue))),
    ])
}

#[pyfunction]
#[pyo3(signature = (motifs, min_overlap, rc, name, weighted, nthreads))]
fn _merge_motifs(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    min_overlap: usize,
    rc: bool,
    name: Option<String>,
    weighted: bool,
    nthreads: usize,
) -> PyResult<(PyMotif, Vec<usize>)> {
    let motifs = unwrap(&motifs);
    let opts = MergeOptions {
        min_overlap,
        rc,
        name,
        weighted,
        nthreads,
    };
    py.detach(|| merge::merge_motifs_detailed(&motifs, &opts))
        .map(|m| (PyMotif::from(m.motif), m.used))
        .map_err(err)
}

#[pyfunction]
#[pyo3(signature = (motifs, qvalue, null, min_overlap, rc, bkg, weighted, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _merge_similar(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    qvalue: f64,
    null: &str,
    min_overlap: usize,
    rc: bool,
    bkg: Option<Vec<f64>>,
    weighted: bool,
    nthreads: usize,
) -> PyResult<(Vec<usize>, Vec<PyMotif>)> {
    let motifs = unwrap(&motifs);
    let opts = MergeSimilarOptions {
        qvalue,
        null: null_model(null)?,
        min_overlap,
        rc,
        bkg: bkg4(bkg)?,
        weighted,
        nthreads,
    };
    let res = py
        .detach(|| merge::merge_similar(&motifs, &opts))
        .map_err(err)?;
    Ok((
        res.clusters,
        res.motifs.into_iter().map(PyMotif::from).collect(),
    ))
}

fn tree_dict<'py>(py: Python<'py>, tree: &Tree, labels: &[String]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "merge",
        tree.merge.iter().map(|m| m.to_vec()).collect::<Vec<_>>(),
    )?;
    d.set_item("height", &tree.height)?;
    d.set_item("order", &tree.order)?;
    d.set_item("newick", tree.newick(labels))?;
    d.set_item(
        "linkage",
        tree.linkage()
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
    )?;
    Ok(d)
}

fn method(m: &str) -> PyResult<Linkage> {
    m.parse::<Linkage>().map_err(err)
}

#[pyfunction]
fn _hclust<'py>(
    py: Python<'py>,
    dist: Vec<Vec<f64>>,
    method: &str,
    labels: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = self::method(method)?;
    let tree = py.detach(|| cluster::hclust(&dist, m)).map_err(err)?;
    tree_dict(py, &tree, &labels)
}

#[pyfunction]
#[pyo3(signature = (motifs, min_overlap, rc, method, labels, nthreads))]
fn _motif_tree<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    min_overlap: usize,
    rc: bool,
    method: &str,
    labels: Vec<String>,
    nthreads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let motifs = unwrap(&motifs);
    let opts = TreeOptions {
        min_overlap,
        rc,
        method: self::method(method)?,
        nthreads,
    };
    let tree = py
        .detach(|| cluster::motif_tree(&motifs, &opts))
        .map_err(err)?;
    tree_dict(py, &tree, &labels)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_compare_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_compare_table, m)?)?;
    m.add_function(wrap_pyfunction!(_merge_motifs, m)?)?;
    m.add_function(wrap_pyfunction!(_merge_similar, m)?)?;
    m.add_function(wrap_pyfunction!(_hclust, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_tree, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_pvalue, m)?)?;
    m.add_function(wrap_pyfunction!(_min_pvalues, m)?)?;
    m.add_function(wrap_pyfunction!(_scan_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(_scan_packed, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_matches, m)?)?;
    m.add_function(wrap_pyfunction!(_scan_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(_dedup, m)?)?;
    Ok(())
}
