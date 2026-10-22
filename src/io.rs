//! Bindings for the motif file readers and writers. Python handles opening,
//! skipping and appending to files; these functions work on text.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBool;
use universalmotif::Motif;
use universalmotif::io::{cisbp, fasta, homer, jaspar, matrix, meme, transfac, uniprobe};

use crate::{PyMotif, err, parse_alphabet, parse_type};

fn wrap(r: universalmotif::Result<Vec<Motif>>) -> PyResult<Vec<PyMotif>> {
    r.map(|v| v.into_iter().map(PyMotif::from).collect())
        .map_err(err)
}

fn unwrap(motifs: &[PyRef<'_, PyMotif>]) -> Vec<Motif> {
    motifs.iter().map(|m| m.inner.clone()).collect()
}

fn headers(h: &Bound<'_, PyAny>) -> PyResult<matrix::Headers> {
    if let Ok(b) = h.cast::<PyBool>() {
        Ok(if b.is_true() {
            matrix::Headers::FirstLine
        } else {
            matrix::Headers::None
        })
    } else if let Ok(s) = h.extract::<String>() {
        Ok(matrix::Headers::Prefix(s))
    } else {
        Err(PyTypeError::new_err(
            "headers must be True, False or a string",
        ))
    }
}

fn positions(p: &str) -> PyResult<matrix::Positions> {
    match p {
        "columns" => Ok(matrix::Positions::Columns),
        "rows" => Ok(matrix::Positions::Rows),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "positions must be 'columns' or 'rows'",
        )),
    }
}

#[pyfunction]
fn _read_jaspar(text: &str) -> PyResult<Vec<PyMotif>> {
    wrap(jaspar::read(text))
}

#[pyfunction]
fn _read_homer(text: &str) -> PyResult<Vec<PyMotif>> {
    wrap(homer::read(text))
}

#[pyfunction]
fn _read_transfac(text: &str) -> PyResult<Vec<PyMotif>> {
    wrap(transfac::read(text))
}

#[pyfunction]
fn _read_cisbp(text: &str) -> PyResult<Vec<PyMotif>> {
    wrap(cisbp::read(text))
}

#[pyfunction]
fn _read_uniprobe(text: &str) -> PyResult<Vec<PyMotif>> {
    wrap(uniprobe::read(text))
}

#[pyfunction]
fn _read_meme(text: &str, cwm: bool) -> PyResult<Vec<PyMotif>> {
    wrap(meme::read(text, &meme::ReadOptions { cwm }))
}

type MotifSites = (String, Vec<(String, String)>);

/// `[(motif, [(sequence, site), ...]), ...]`
#[pyfunction]
fn _read_meme_sites(text: &str) -> PyResult<Vec<MotifSites>> {
    let s = meme::read_sites(text).map_err(err)?;
    Ok(s.into_iter()
        .map(|m| {
            let sites = m.sites.into_iter().map(|s| (s.sequence, s.site)).collect();
            (m.motif, sites)
        })
        .collect())
}

type SiteRow = (String, Option<String>, f64, f64, String);

/// `[(motif, [(sequence, strand, position, pvalue, site), ...]), ...]`
#[pyfunction]
fn _read_meme_site_pvalues(text: &str) -> PyResult<Vec<(String, Vec<SiteRow>)>> {
    let s = meme::read_site_pvalues(text).map_err(err)?;
    Ok(s.into_iter()
        .map(|m| {
            let rows = m
                .sites
                .into_iter()
                .map(|s| {
                    let strand = s.strand.map(|c| c.to_string());
                    (s.sequence, strand, s.position, s.pvalue, s.site)
                })
                .collect();
            (m.motif, rows)
        })
        .collect())
}

/// `[(sequence, pvalue, diagram), ...]`
#[pyfunction]
fn _read_meme_combined(text: &str) -> PyResult<Vec<(String, f64, String)>> {
    let s = meme::read_combined(text).map_err(err)?;
    Ok(s.into_iter()
        .map(|c| (c.sequence, c.pvalue, c.diagram))
        .collect())
}

#[pyfunction]
#[pyo3(signature = (text, positions, alphabet, r#type, sep, headers, rownames, comment))]
#[allow(clippy::too_many_arguments)]
fn _read_matrix(
    text: &str,
    positions: &str,
    alphabet: &str,
    r#type: Option<&str>,
    sep: Option<String>,
    headers: &Bound<'_, PyAny>,
    rownames: bool,
    comment: Option<String>,
) -> PyResult<Vec<PyMotif>> {
    let opts = matrix::ReadOptions {
        positions: self::positions(positions)?,
        alphabet: parse_alphabet(alphabet)?,
        motif_type: r#type.map(parse_type).transpose()?,
        sep,
        headers: self::headers(headers)?,
        rownames,
        comment,
    };
    wrap(matrix::read(text, &opts))
}

#[pyfunction]
#[pyo3(signature = (motifs, version, bkg, strand, cwm, header))]
fn _write_meme(
    motifs: Vec<PyRef<'_, PyMotif>>,
    version: u32,
    bkg: Option<Vec<f64>>,
    strand: Option<String>,
    cwm: bool,
    header: bool,
) -> PyResult<String> {
    let opts = meme::WriteOptions {
        version,
        bkg,
        strand,
        cwm,
        header,
    };
    meme::write(&unwrap(&motifs), &opts).map_err(err)
}

#[pyfunction]
fn _write_jaspar(motifs: Vec<PyRef<'_, PyMotif>>) -> PyResult<String> {
    jaspar::write(&unwrap(&motifs)).map_err(err)
}

#[pyfunction]
fn _write_homer(
    motifs: Vec<PyRef<'_, PyMotif>>,
    threshold: Vec<f64>,
    threshold_type: &str,
) -> PyResult<String> {
    let t = match threshold_type {
        "logodds" => homer::Threshold::Logodds(threshold),
        "logodds_abs" => homer::Threshold::LogoddsAbs(threshold),
        "pvalue" => homer::Threshold::Pvalue(threshold),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "threshold_type must be 'logodds', 'logodds_abs' or 'pvalue'",
            ));
        }
    };
    homer::write(&unwrap(&motifs), &t).map_err(err)
}

#[pyfunction]
fn _write_transfac(
    motifs: Vec<PyRef<'_, PyMotif>>,
    name_tag: String,
    altname_tag: String,
) -> PyResult<String> {
    let opts = transfac::WriteOptions {
        name_tag,
        altname_tag,
    };
    transfac::write(&unwrap(&motifs), &opts).map_err(err)
}

#[pyfunction]
fn _write_cisbp(motifs: Vec<PyRef<'_, PyMotif>>) -> PyResult<String> {
    cisbp::write(&unwrap(&motifs)).map_err(err)
}

#[pyfunction]
fn _write_uniprobe(motifs: Vec<PyRef<'_, PyMotif>>) -> PyResult<String> {
    uniprobe::write(&unwrap(&motifs)).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (motifs, positions, rownames, r#type, sep, headers, digits))]
fn _write_matrix(
    motifs: Vec<PyRef<'_, PyMotif>>,
    positions: &str,
    rownames: bool,
    r#type: Option<&str>,
    sep: Option<String>,
    headers: &Bound<'_, PyAny>,
    digits: usize,
) -> PyResult<String> {
    let opts = matrix::WriteOptions {
        positions: self::positions(positions)?,
        rownames,
        motif_type: r#type.map(parse_type).transpose()?,
        sep,
        headers: self::headers(headers)?,
        digits,
    };
    matrix::write(&unwrap(&motifs), &opts).map_err(err)
}

#[pyfunction]
fn _read_fasta(text: &str) -> PyResult<Vec<(String, String)>> {
    fasta::read(text).map_err(err)
}

#[pyfunction]
fn _write_fasta(records: Vec<(String, String)>, width: usize) -> String {
    fasta::write(&records, width)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_read_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(_write_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(_read_jaspar, m)?)?;
    m.add_function(wrap_pyfunction!(_read_homer, m)?)?;
    m.add_function(wrap_pyfunction!(_read_transfac, m)?)?;
    m.add_function(wrap_pyfunction!(_read_cisbp, m)?)?;
    m.add_function(wrap_pyfunction!(_read_uniprobe, m)?)?;
    m.add_function(wrap_pyfunction!(_read_meme, m)?)?;
    m.add_function(wrap_pyfunction!(_read_meme_sites, m)?)?;
    m.add_function(wrap_pyfunction!(_read_meme_site_pvalues, m)?)?;
    m.add_function(wrap_pyfunction!(_read_meme_combined, m)?)?;
    m.add_function(wrap_pyfunction!(_read_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_write_meme, m)?)?;
    m.add_function(wrap_pyfunction!(_write_jaspar, m)?)?;
    m.add_function(wrap_pyfunction!(_write_homer, m)?)?;
    m.add_function(wrap_pyfunction!(_write_transfac, m)?)?;
    m.add_function(wrap_pyfunction!(_write_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_write_cisbp, m)?)?;
    m.add_function(wrap_pyfunction!(_write_uniprobe, m)?)?;
    Ok(())
}
