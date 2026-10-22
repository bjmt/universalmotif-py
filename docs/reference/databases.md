# Databases and Biopython

Motifs from public databases, and conversion to and from Biopython.

The database functions live in the `universalmotif.databases` module, used
as `um.databases.jaspar()` and so on. They download what they need the first
time and keep it in a cache, so later calls work offline. The conversion
functions need the `biopython` extra:

```sh
pip install universalmotif[biopython]
```

## Databases

::: universalmotif.databases.jaspar

::: universalmotif.databases.hocomoco

::: universalmotif.databases.cisbp

::: universalmotif.databases.read_cisbp_archive

::: universalmotif.databases.cache_dir

::: universalmotif.databases.clear_cache

## Biopython

::: universalmotif.to_biopython

::: universalmotif.from_biopython
