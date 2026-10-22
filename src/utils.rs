//! Bindings for column conversions, scoring helpers, column metrics, gap and
//! alphabet operations, masking and small helpers.

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use universalmotif::io::meme::{AlphabetDefinition, write_alphabet};
use universalmotif::metrics::{Average, ColumnMetric};
use universalmotif::scoring::{self, ThresholdType};
use universalmotif::{convert, mask, metrics, ops, pvalue};

use crate::{PyMotif, err, parse_alphabet};

#[pyfunction]
fn _pcm_to_ppm(pos: Vec<f64>, pseudocount: f64) -> Vec<f64> {
    convert::pcm_to_ppm(&pos, pseudocount)
}

#[pyfunction]
fn _ppm_to_pcm(pos: Vec<f64>, nsites: f64) -> Vec<f64> {
    convert::ppm_to_pcm(&pos, nsites)
}

#[pyfunction]
fn _ppm_to_pwm(pos: Vec<f64>, bkg: Vec<f64>, pseudocount: f64, nsites: f64) -> Vec<f64> {
    convert::ppm_to_pwm(&pos, &bkg, pseudocount, nsites)
}

#[pyfunction]
fn _pwm_to_ppm(pos: Vec<f64>, bkg: Vec<f64>) -> Vec<f64> {
    convert::pwm_to_ppm(&pos, &bkg)
}

#[pyfunction]
fn _ppm_to_icm(pos: Vec<f64>, bkg: Vec<f64>, relative_entropy: bool) -> Vec<f64> {
    convert::ppm_to_icm(&pos, &bkg, relative_entropy)
}

#[pyfunction]
fn _icm_to_ppm(pos: Vec<f64>) -> Vec<f64> {
    convert::icm_to_ppm(&pos)
}

#[pyfunction]
fn _motif_range(motif: PyRef<'_, PyMotif>, allow_nonfinite: bool) -> PyResult<(f64, f64)> {
    scoring::motif_range(&motif.inner, allow_nonfinite).map_err(err)
}

#[pyfunction]
fn _motif_score(
    motif: PyRef<'_, PyMotif>,
    thresholds: Vec<f64>,
    threshold_type: &str,
    allow_nonfinite: bool,
) -> PyResult<Vec<f64>> {
    let t = ThresholdType::parse(threshold_type).map_err(err)?;
    scoring::motif_score(&motif.inner, &thresholds, t, allow_nonfinite).map_err(err)
}

#[pyfunction]
fn _score_match(
    motif: PyRef<'_, PyMotif>,
    words: Vec<PyBackedStr>,
    allow_nonfinite: bool,
) -> PyResult<Vec<f64>> {
    scoring::score_match(&motif.inner, &words, allow_nonfinite).map_err(err)
}

#[pyfunction]
fn _get_scores(
    py: Python<'_>,
    motif: PyRef<'_, PyMotif>,
    allow_nonfinite: bool,
) -> PyResult<Vec<f64>> {
    let m = motif.inner.clone();
    py.detach(|| scoring::get_scores(&m, allow_nonfinite))
        .map_err(err)
}

#[pyfunction]
fn _get_matches(
    py: Python<'_>,
    motif: PyRef<'_, PyMotif>,
    score: f64,
    allow_nonfinite: bool,
) -> PyResult<Vec<String>> {
    let m = motif.inner.clone();
    py.detach(|| scoring::get_matches(&m, score, allow_nonfinite))
        .map_err(err)
}

#[pyfunction]
fn _prob_match(
    motif: PyRef<'_, PyMotif>,
    words: Vec<PyBackedStr>,
    allow_zero: bool,
) -> PyResult<Vec<f64>> {
    scoring::prob_match(&motif.inner, &words, allow_zero).map_err(err)
}

#[pyfunction]
fn _prob_match_bkg(bkg: Vec<(char, f64)>, words: Vec<PyBackedStr>) -> PyResult<Vec<f64>> {
    scoring::prob_match_bkg(&bkg, &words).map_err(err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _compare_columns(
    x: Vec<f64>,
    y: Vec<f64>,
    method: &str,
    bkg1: Vec<f64>,
    bkg2: Vec<f64>,
    nsites1: f64,
    nsites2: f64,
) -> PyResult<f64> {
    let metric = ColumnMetric::parse(method).map_err(err)?;
    metrics::compare_columns(&x, &y, metric, &bkg1, &bkg2, nsites1, nsites2).map_err(err)
}

#[pyfunction]
fn _average_ic(motifs: Vec<PyRef<'_, PyMotif>>, average: &str) -> PyResult<Vec<f64>> {
    let how = Average::parse(average).map_err(err)?;
    let motifs: Vec<_> = motifs.iter().map(|m| m.inner.clone()).collect();
    metrics::average_ic(&motifs, how).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (motif, gaploc, mingap, maxgap))]
fn _add_gap(
    motif: PyRef<'_, PyMotif>,
    gaploc: Option<Vec<usize>>,
    mingap: Vec<usize>,
    maxgap: Vec<usize>,
) -> PyResult<PyMotif> {
    ops::add_gap(&motif.inner, gaploc.as_deref(), &mingap, &maxgap)
        .map(PyMotif::from)
        .map_err(err)
}

#[pyfunction]
fn _ungap(motif: PyRef<'_, PyMotif>, delete: bool) -> PyMotif {
    PyMotif::from(ops::ungap(&motif.inner, delete))
}

#[pyfunction]
fn _switch_alph(motif: PyRef<'_, PyMotif>) -> PyResult<PyMotif> {
    ops::switch_alph(&motif.inner)
        .map(PyMotif::from)
        .map_err(err)
}

#[pyfunction]
fn _mask_seqs(
    py: Python<'_>,
    seqs: Vec<PyBackedStr>,
    pattern: &str,
    rc: bool,
    letter: char,
    alphabet: &str,
) -> PyResult<Vec<String>> {
    let alphabet = parse_alphabet(alphabet)?;
    py.detach(|| mask::mask_seqs(&seqs, pattern, rc, letter, &alphabet))
        .map_err(err)
}

#[pyfunction]
fn _mask_ranges(seq: PyBackedStr, ranges: Vec<(i64, i64)>, letter: char) -> PyResult<String> {
    mask::mask_ranges(&seq, &ranges, letter).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (core, complements, ambiguity, like, name, letter_names, colours))]
fn _meme_alph(
    core: String,
    complements: Option<String>,
    ambiguity: Vec<(String, String)>,
    like: Option<String>,
    name: Option<String>,
    letter_names: Vec<(char, String)>,
    colours: Vec<(char, String)>,
) -> PyResult<String> {
    write_alphabet(&AlphabetDefinition {
        core,
        complements,
        ambiguity,
        like,
        name,
        letter_names,
        colours,
    })
    .map_err(err)
}

#[pyfunction]
fn _log_string_pval(pval: &str) -> PyResult<f64> {
    pvalue::log_pvalue_text(pval).map_err(err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_pcm_to_ppm, m)?)?;
    m.add_function(wrap_pyfunction!(_ppm_to_pcm, m)?)?;
    m.add_function(wrap_pyfunction!(_ppm_to_pwm, m)?)?;
    m.add_function(wrap_pyfunction!(_pwm_to_ppm, m)?)?;
    m.add_function(wrap_pyfunction!(_ppm_to_icm, m)?)?;
    m.add_function(wrap_pyfunction!(_icm_to_ppm, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_range, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_score, m)?)?;
    m.add_function(wrap_pyfunction!(_score_match, m)?)?;
    m.add_function(wrap_pyfunction!(_get_scores, m)?)?;
    m.add_function(wrap_pyfunction!(_get_matches, m)?)?;
    m.add_function(wrap_pyfunction!(_prob_match, m)?)?;
    m.add_function(wrap_pyfunction!(_prob_match_bkg, m)?)?;
    m.add_function(wrap_pyfunction!(_compare_columns, m)?)?;
    m.add_function(wrap_pyfunction!(_average_ic, m)?)?;
    m.add_function(wrap_pyfunction!(_add_gap, m)?)?;
    m.add_function(wrap_pyfunction!(_ungap, m)?)?;
    m.add_function(wrap_pyfunction!(_switch_alph, m)?)?;
    m.add_function(wrap_pyfunction!(_mask_seqs, m)?)?;
    m.add_function(wrap_pyfunction!(_mask_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(_meme_alph, m)?)?;
    m.add_function(wrap_pyfunction!(_log_string_pval, m)?)?;
    Ok(())
}
