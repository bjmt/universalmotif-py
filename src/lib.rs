//! Python bindings for the universalmotif Rust crate. This layer only
//! converts between Python and Rust types; the motif logic lives in the crate,
//! and the user-facing Python
//! API (argument handling, lists of motifs) lives in `python/universalmotif`.

use numpy::ndarray::{Array2, ShapeBuilder};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use universalmotif::create::{self, CreateOptions};
use universalmotif::ops::{self, TrimFrom};
use universalmotif::{Alphabet, GapInfo, Matrix, Motif, MotifParts, MotifType, Strand, convert};

mod analysis;
mod arrow;
mod enrichment;
mod genome;
mod genomic;
mod io;
mod packed;
mod plot;
mod sequences;
mod utils;

pub(crate) fn err(e: universalmotif::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

pub(crate) fn parse_type(s: &str) -> PyResult<MotifType> {
    s.parse::<MotifType>().map_err(err)
}

pub(crate) fn parse_alphabet(s: &str) -> PyResult<Alphabet> {
    s.parse::<Alphabet>().map_err(err)
}

fn parse_strand(s: &str) -> PyResult<Strand> {
    s.parse::<Strand>().map_err(err)
}

fn one_char(s: &str, what: &str) -> PyResult<char> {
    let mut it = s.chars();
    match (it.next(), it.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(PyValueError::new_err(format!(
            "{what} must be a single character, got {s:?}"
        ))),
    }
}

/// Read background frequencies given either as a sequence (one value per
/// letter, in alphabet order) or as a mapping from letters to values.
/// Mapping keys longer than one letter (higher-order backgrounds) are
/// ignored.
pub(crate) fn parse_bkg(obj: &Bound<'_, PyAny>, alphabet: &Alphabet) -> PyResult<Vec<f64>> {
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut out = vec![None; alphabet.len()];
        for (k, v) in d.iter() {
            let k: String = k.extract()?;
            let mut chars = k.chars();
            let (Some(c), None) = (chars.next(), chars.next()) else {
                continue;
            };
            match alphabet.index_of(c) {
                Some(i) => out[i] = Some(v.extract::<f64>()?),
                None => {
                    return Err(PyValueError::new_err(format!(
                        "bkg letter {k:?} is not in the alphabet ({alphabet})"
                    )));
                }
            }
        }
        out.into_iter()
            .collect::<Option<Vec<f64>>>()
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "bkg must contain a value for every letter of the alphabet ({alphabet})"
                ))
            })
    } else {
        let v: Vec<f64> = obj.extract()?;
        if v.len() != alphabet.len() {
            return Err(PyValueError::new_err(format!(
                "bkg has {} values but the alphabet has {} letters",
                v.len(),
                alphabet.len()
            )));
        }
        Ok(v)
    }
}

fn parse_extrainfo(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(String, String)>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        d.iter()
            .map(|(k, v)| Ok((k.extract()?, v.extract()?)))
            .collect()
    } else {
        Ok(obj.extract()?)
    }
}

fn matrix_from_numpy(m: PyReadonlyArray2<'_, f64>) -> PyResult<Matrix> {
    let a = m.as_array();
    let (nrow, ncol) = a.dim();
    let mut data = Vec::with_capacity(nrow * ncol);
    for j in 0..ncol {
        for i in 0..nrow {
            data.push(a[[i, j]]);
        }
    }
    Matrix::from_vec(nrow, ncol, data).map_err(err)
}

fn matrix_to_numpy<'py>(py: Python<'py>, m: &Matrix) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = Array2::from_shape_vec((m.nrow(), m.ncol()).f(), m.as_slice().to_vec())
        .expect("matrix shape matches its data");
    let a = PyArray2::from_owned_array(py, arr);
    a.call_method1("setflags", (false,))?;
    Ok(a)
}

/// A sequence motif.
///
/// Motifs are immutable. Use functions such as `convert_type` or
/// `trim_motifs` to derive new motifs, or `replace()` to change metadata.
#[pyclass(name = "Motif", module = "universalmotif", frozen)]
#[derive(Clone)]
pub struct PyMotif {
    pub(crate) inner: Motif,
}

impl From<Motif> for PyMotif {
    fn from(inner: Motif) -> Self {
        PyMotif { inner }
    }
}

#[pymethods]
impl PyMotif {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn altname(&self) -> Option<&str> {
        self.inner.altname.as_deref()
    }
    #[getter]
    fn family(&self) -> Option<&str> {
        self.inner.family.as_deref()
    }
    #[getter]
    fn organism(&self) -> Option<&str> {
        self.inner.organism.as_deref()
    }
    /// The motif matrix as a read-only array of shape (letters, positions).
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        matrix_to_numpy(py, &self.inner.matrix)
    }
    /// The alphabet name: "DNA", "RNA", "AA", or the letters of a custom
    /// alphabet.
    #[getter]
    fn alphabet(&self) -> String {
        self.inner.alphabet.name()
    }
    /// The row letters of the matrix.
    #[getter]
    fn letters(&self) -> Vec<String> {
        self.inner
            .alphabet
            .letters()
            .iter()
            .map(|c| c.to_string())
            .collect()
    }
    /// The matrix type: "PCM", "PPM", "PWM", "ICM" or "CWM".
    #[getter]
    fn r#type(&self) -> &'static str {
        self.inner.motif_type.as_str()
    }
    #[getter]
    fn icscore(&self) -> f64 {
        self.inner.icscore
    }
    #[getter]
    fn nsites(&self) -> Option<f64> {
        self.inner.nsites
    }
    #[getter]
    fn pseudocount(&self) -> f64 {
        self.inner.pseudocount
    }
    /// Background frequencies, keyed by letter.
    #[getter]
    fn bkg<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (l, b) in self.inner.alphabet.letters().iter().zip(&self.inner.bkg) {
            d.set_item(l.to_string(), b)?;
        }
        Ok(d)
    }
    #[getter]
    fn bkgsites(&self) -> Option<f64> {
        self.inner.bkgsites
    }
    #[getter]
    fn consensus(&self) -> Option<&str> {
        self.inner.consensus.as_deref()
    }
    #[getter]
    fn strand(&self) -> &'static str {
        self.inner.strand.as_str()
    }
    #[getter]
    fn pval(&self) -> Option<f64> {
        self.inner.pval
    }
    #[getter]
    fn qval(&self) -> Option<f64> {
        self.inner.qval
    }
    #[getter]
    fn eval(&self) -> Option<f64> {
        self.inner.eval
    }
    #[getter]
    fn extrainfo<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.inner.extrainfo {
            d.set_item(k, v)?;
        }
        Ok(d)
    }
    #[getter]
    fn isgapped(&self) -> bool {
        self.inner.gap.isgapped
    }
    #[getter]
    fn gaploc(&self) -> Vec<usize> {
        self.inner.gap.gaploc.clone()
    }
    #[getter]
    fn mingap(&self) -> Vec<usize> {
        self.inner.gap.mingap.clone()
    }
    #[getter]
    fn maxgap(&self) -> Vec<usize> {
        self.inner.gap.maxgap.clone()
    }
    /// Number of positions.
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn __len__(&self) -> usize {
        self.inner.width()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string().trim_matches('\n').to_string()
    }

    fn __eq__(&self, other: PyRef<'_, PyMotif>) -> bool {
        self.inner == other.inner
    }

    /// Return a copy with some metadata replaced. The matrix, type,
    /// alphabet, consensus and total IC cannot be changed this way, but the
    /// total IC and consensus are recomputed when the pseudocount,
    /// background or nsites change.
    #[pyo3(signature = (**changes))]
    fn replace(&self, changes: Option<&Bound<'_, PyDict>>) -> PyResult<PyMotif> {
        let mut p = self.inner.clone().into_parts();
        let mut refresh = false;
        if let Some(changes) = changes {
            for (k, v) in changes.iter() {
                let k: String = k.extract()?;
                refresh |= matches!(k.as_str(), "pseudocount" | "bkg" | "nsites");
                match k.as_str() {
                    "name" => p.name = v.extract()?,
                    "altname" => p.altname = v.extract()?,
                    "family" => p.family = v.extract()?,
                    "organism" => p.organism = v.extract()?,
                    "nsites" => p.nsites = v.extract()?,
                    "pseudocount" => p.pseudocount = v.extract()?,
                    "bkg" => p.bkg = parse_bkg(&v, &p.alphabet)?,
                    "bkgsites" => p.bkgsites = v.extract()?,
                    "strand" => p.strand = parse_strand(&v.extract::<String>()?)?,
                    "pval" => p.pval = v.extract()?,
                    "qval" => p.qval = v.extract()?,
                    "eval" => p.eval = v.extract()?,
                    "extrainfo" => p.extrainfo = parse_extrainfo(&v)?,
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "replace() cannot change {k:?}"
                        )));
                    }
                }
            }
        }
        if refresh {
            p.recompute_derived();
        }
        Motif::from_parts(p).map(PyMotif::from).map_err(err)
    }

    /// Every field of the motif as plain Python objects, as used for
    /// pickling.
    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = &*self.inner;
        let d = PyDict::new(py);
        d.set_item("name", &p.name)?;
        d.set_item("altname", &p.altname)?;
        d.set_item("family", &p.family)?;
        d.set_item("organism", &p.organism)?;
        d.set_item("alphabet", p.alphabet.name())?;
        d.set_item("type", p.motif_type.as_str())?;
        d.set_item("icscore", p.icscore)?;
        d.set_item("nsites", p.nsites)?;
        d.set_item("pseudocount", p.pseudocount)?;
        d.set_item("bkg", &p.bkg)?;
        d.set_item("bkgsites", p.bkgsites)?;
        d.set_item("consensus", &p.consensus)?;
        d.set_item("strand", p.strand.as_str())?;
        d.set_item("pval", p.pval)?;
        d.set_item("qval", p.qval)?;
        d.set_item("eval", p.eval)?;
        d.set_item("extrainfo", &p.extrainfo)?;
        let gap = PyDict::new(py);
        gap.set_item("isgapped", p.gap.isgapped)?;
        gap.set_item("gaploc", &p.gap.gaploc)?;
        gap.set_item("mingap", &p.gap.mingap)?;
        gap.set_item("maxgap", &p.gap.maxgap)?;
        d.set_item("gap", gap)?;
        let cols: Vec<Vec<f64>> = p.matrix.columns().map(|c| c.to_vec()).collect();
        d.set_item("matrix", cols)?;
        Ok(d)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let f = py
            .import("universalmotif._core")?
            .getattr("_motif_from_state")?;
        let args = PyTuple::new(py, [self._state(py)?])?;
        PyTuple::new(py, [f, args.into_any()])
    }
}

fn item<'py>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(k)?
        .ok_or_else(|| PyValueError::new_err(format!("motif state is missing {k:?}")))
}

/// Rebuild a motif from the dictionary produced by `Motif._state()`, without
/// recomputing anything.
#[pyfunction]
fn _motif_from_state(state: &Bound<'_, PyDict>) -> PyResult<PyMotif> {
    let gap = item(state, "gap")?;
    let gap = gap.cast::<PyDict>()?;
    let cols: Vec<Vec<f64>> = item(state, "matrix")?.extract()?;
    let parts = MotifParts {
        name: item(state, "name")?.extract()?,
        altname: item(state, "altname")?.extract()?,
        family: item(state, "family")?.extract()?,
        organism: item(state, "organism")?.extract()?,
        matrix: Matrix::from_columns(&cols).map_err(err)?,
        alphabet: parse_alphabet(&item(state, "alphabet")?.extract::<String>()?)?,
        motif_type: parse_type(&item(state, "type")?.extract::<String>()?)?,
        icscore: item(state, "icscore")?.extract()?,
        nsites: item(state, "nsites")?.extract()?,
        pseudocount: item(state, "pseudocount")?.extract()?,
        bkg: item(state, "bkg")?.extract()?,
        bkgsites: item(state, "bkgsites")?.extract()?,
        consensus: item(state, "consensus")?.extract()?,
        strand: parse_strand(&item(state, "strand")?.extract::<String>()?)?,
        pval: item(state, "pval")?.extract()?,
        qval: item(state, "qval")?.extract()?,
        eval: item(state, "eval")?.extract()?,
        extrainfo: parse_extrainfo(&item(state, "extrainfo")?)?,
        gap: GapInfo {
            isgapped: item(gap, "isgapped")?.extract()?,
            gaploc: item(gap, "gaploc")?.extract()?,
            mingap: item(gap, "mingap")?.extract()?,
            maxgap: item(gap, "maxgap")?.extract()?,
        },
    };
    Motif::from_parts(parts).map(PyMotif::from).map_err(err)
}

/// Build `CreateOptions` from keyword arguments. `bkg_alphabet` is only
/// called when a background is given, to put its values in letter order.
fn create_opts(
    kwargs: Option<&Bound<'_, PyDict>>,
    bkg_alphabet: impl FnOnce() -> PyResult<Alphabet>,
) -> PyResult<CreateOptions> {
    let mut o = CreateOptions::default();
    let Some(kwargs) = kwargs else {
        return Ok(o);
    };
    let mut bkg = None;
    for (k, v) in kwargs.iter() {
        let k: String = k.extract()?;
        if v.is_none() {
            continue;
        }
        match k.as_str() {
            "type" => o.motif_type = parse_type(&v.extract::<String>()?)?,
            "name" => o.name = v.extract()?,
            "pseudocount" => o.pseudocount = v.extract()?,
            "nsites" => o.nsites = Some(v.extract()?),
            "bkg" => bkg = Some(v),
            "altname" => o.altname = Some(v.extract()?),
            "family" => o.family = Some(v.extract()?),
            "organism" => o.organism = Some(v.extract()?),
            "bkgsites" => o.bkgsites = Some(v.extract()?),
            "strand" => o.strand = parse_strand(&v.extract::<String>()?)?,
            "pval" => o.pval = Some(v.extract()?),
            "qval" => o.qval = Some(v.extract()?),
            "eval" => o.eval = Some(v.extract()?),
            "extrainfo" => o.extrainfo = parse_extrainfo(&v)?,
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unexpected keyword argument {k:?}"
                )));
            }
        }
    }
    if let Some(b) = bkg {
        o.bkg = Some(parse_bkg(&b, &bkg_alphabet()?)?);
    }
    Ok(o)
}

#[pyfunction]
#[pyo3(signature = (matrix, letters=None, alphabet=None, matrix_type=None, **kwargs))]
fn _create_from_matrix(
    matrix: PyReadonlyArray2<'_, f64>,
    letters: Option<Vec<String>>,
    alphabet: Option<&str>,
    matrix_type: Option<&str>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyMotif> {
    let matrix = matrix_from_numpy(matrix)?;
    let letters: Option<Vec<char>> = letters
        .map(|l| l.iter().map(|s| one_char(s, "a row letter")).collect())
        .transpose()?;
    let alphabet = alphabet.map(parse_alphabet).transpose()?;
    let opts = create_opts(kwargs, || match (&alphabet, &letters) {
        (Some(a), _) => Ok(a.clone()),
        (None, Some(l)) => {
            let mut l = l.clone();
            l.sort_unstable();
            create::guess_matrix_alphabet(&l).map_err(err)
        }
        (None, None) => Err(PyValueError::new_err(
            "please provide an alphabet or row letters",
        )),
    })?;
    match matrix_type {
        Some(t) => {
            create::from_typed_matrix(matrix, letters.as_deref(), alphabet, parse_type(t)?, &opts)
        }
        None => create::from_matrix(matrix, letters.as_deref(), alphabet, &opts),
    }
    .map(PyMotif::from)
    .map_err(err)
}

#[pyfunction]
#[pyo3(signature = (strings, alphabet=None, **kwargs))]
fn _create_from_strings(
    strings: Vec<String>,
    alphabet: Option<&str>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyMotif> {
    let seqs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let alphabet = alphabet.map(parse_alphabet).transpose()?;
    let opts = create_opts(kwargs, || match &alphabet {
        Some(a) => Ok(a.clone()),
        None => create::guess_strings_alphabet(&seqs).map_err(err),
    })?;
    create::from_strings(&seqs, alphabet, &opts)
        .map(PyMotif::from)
        .map_err(err)
}

#[pyfunction]
#[pyo3(signature = (width, alphabet="DNA", seed=None, **kwargs))]
fn _create_random(
    width: usize,
    alphabet: &str,
    seed: Option<u64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyMotif> {
    let alphabet = parse_alphabet(alphabet)?;
    let opts = create_opts(kwargs, || Ok(alphabet.clone()))?;
    let seed = seed.unwrap_or_else(rand::random);
    create::random(width, alphabet, &opts, seed)
        .map(PyMotif::from)
        .map_err(err)
}

#[pyfunction]
#[pyo3(signature = (motif, r#type, pseudocount=None, relative_entropy=false))]
fn _convert_type(
    motif: PyRef<'_, PyMotif>,
    r#type: &str,
    pseudocount: Option<f64>,
    relative_entropy: bool,
) -> PyResult<PyMotif> {
    convert::convert_type(
        &motif.inner,
        parse_type(r#type)?,
        pseudocount,
        relative_entropy,
    )
    .map(PyMotif::from)
    .map_err(err)
}

#[pyfunction]
fn _motif_rc(motif: PyRef<'_, PyMotif>, ignore_alphabet: bool) -> PyResult<PyMotif> {
    ops::reverse_complement(&motif.inner, ignore_alphabet)
        .map(PyMotif::from)
        .map_err(err)
}

#[pyfunction]
fn _trim_motif(
    motif: PyRef<'_, PyMotif>,
    min_ic: f64,
    trim_from: &str,
) -> PyResult<Option<PyMotif>> {
    let tf = trim_from.parse::<TrimFrom>().map_err(err)?;
    ops::trim_motif(&motif.inner, min_ic, tf)
        .map(|m| m.map(PyMotif::from))
        .map_err(err)
}

#[pyfunction]
#[pyo3(signature = (motif, trim_threshold, abs_threshold, trim_from))]
fn _trim_cwm(
    motif: PyRef<'_, PyMotif>,
    trim_threshold: f64,
    abs_threshold: Option<f64>,
    trim_from: &str,
) -> PyResult<Option<PyMotif>> {
    let tf = trim_from.parse::<TrimFrom>().map_err(err)?;
    ops::trim_cwm(&motif.inner, trim_threshold, abs_threshold, tf)
        .map(|m| m.map(PyMotif::from))
        .map_err(err)
}

#[pyfunction]
fn _round_motif(motif: PyRef<'_, PyMotif>, pct_tolerance: f64) -> PyResult<PyMotif> {
    ops::round_motif(&motif.inner, pct_tolerance)
        .map(PyMotif::from)
        .map_err(err)
}

#[pyfunction]
fn _normalize(motif: PyRef<'_, PyMotif>) -> PyResult<PyMotif> {
    ops::normalize(&motif.inner).map(PyMotif::from).map_err(err)
}

#[pyfunction]
fn _position_icscore(
    position: Vec<f64>,
    bkg: Vec<f64>,
    r#type: &str,
    pseudocount: f64,
    nsites: f64,
    relative_entropy: bool,
) -> PyResult<f64> {
    Ok(convert::position_icscore(
        &position,
        &bkg,
        parse_type(r#type)?,
        pseudocount,
        nsites,
        relative_entropy,
    ))
}

#[pyfunction]
fn _get_consensus(
    position: Vec<f64>,
    rna: bool,
    r#type: &str,
    pseudocount: f64,
) -> PyResult<String> {
    if position.len() != 4 {
        return Err(PyValueError::new_err(
            "position must have four values (one per DNA/RNA letter)",
        ));
    }
    Ok(convert::consensus_letter(&position, rna, parse_type(r#type)?, pseudocount).to_string())
}

#[pyfunction]
fn _get_consensus_aa(position: Vec<f64>, r#type: &str, pseudocount: f64) -> PyResult<String> {
    if position.len() != 20 {
        return Err(PyValueError::new_err(
            "position must have twenty values (one per amino acid)",
        ));
    }
    Ok(convert::consensus_letter_aa(&position, parse_type(r#type)?, pseudocount).to_string())
}

#[pyfunction]
fn _consensus_to_ppm(letter: &str) -> PyResult<Vec<f64>> {
    let c = one_char(letter, "letter")?;
    convert::consensus_to_ppm(c)
        .map(|p| p.to_vec())
        .map_err(err)
}

#[pyfunction]
fn _consensus_to_ppm_aa(letter: &str) -> PyResult<Vec<f64>> {
    let c = one_char(letter, "letter")?;
    convert::consensus_to_ppm_aa(c)
        .map(|p| p.to_vec())
        .map_err(err)
}

/// Summary columns for a list of motifs, as a dict of lists.
#[pyfunction]
fn _summarize<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
) -> PyResult<Bound<'py, PyDict>> {
    let ms: Vec<&Motif> = motifs.iter().map(|m| &m.inner).collect();
    let strs = |f: fn(&Motif) -> Option<String>| ms.iter().map(|m| f(m)).collect::<Vec<_>>();
    let nums = |f: fn(&Motif) -> Option<f64>| ms.iter().map(|m| f(m)).collect::<Vec<_>>();
    let d = PyDict::new(py);
    d.set_item("name", strs(|m| Some(m.name.clone())))?;
    d.set_item("altname", strs(|m| m.altname.clone()))?;
    d.set_item("family", strs(|m| m.family.clone()))?;
    d.set_item("organism", strs(|m| m.organism.clone()))?;
    d.set_item("consensus", strs(|m| m.consensus.clone()))?;
    d.set_item("alphabet", strs(|m| Some(m.alphabet.name())))?;
    d.set_item("strand", strs(|m| Some(m.strand.to_string())))?;
    d.set_item("icscore", nums(|m| Some(m.icscore)))?;
    d.set_item("nsites", nums(|m| m.nsites))?;
    d.set_item("bkgsites", nums(|m| m.bkgsites))?;
    d.set_item("pval", nums(|m| m.pval))?;
    d.set_item("qval", nums(|m| m.qval))?;
    d.set_item("eval", nums(|m| m.eval))?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMotif>()?;
    m.add_function(wrap_pyfunction!(_motif_from_state, m)?)?;
    m.add_function(wrap_pyfunction!(_create_from_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_create_from_strings, m)?)?;
    m.add_function(wrap_pyfunction!(_create_random, m)?)?;
    m.add_function(wrap_pyfunction!(_convert_type, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_rc, m)?)?;
    m.add_function(wrap_pyfunction!(_trim_motif, m)?)?;
    m.add_function(wrap_pyfunction!(_trim_cwm, m)?)?;
    m.add_function(wrap_pyfunction!(_round_motif, m)?)?;
    m.add_function(wrap_pyfunction!(_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(_position_icscore, m)?)?;
    m.add_function(wrap_pyfunction!(_get_consensus, m)?)?;
    m.add_function(wrap_pyfunction!(_get_consensus_aa, m)?)?;
    m.add_function(wrap_pyfunction!(_consensus_to_ppm, m)?)?;
    m.add_function(wrap_pyfunction!(_consensus_to_ppm_aa, m)?)?;
    m.add_function(wrap_pyfunction!(_summarize, m)?)?;
    io::register(m)?;
    analysis::register(m)?;
    sequences::register(m)?;
    enrichment::register(m)?;
    plot::register(m)?;
    genomic::register(m)?;
    packed::register(m)?;
    genome::register(m)?;
    utils::register(m)?;
    Ok(())
}
