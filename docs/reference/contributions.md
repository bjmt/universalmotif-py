# Contribution scores

Reading a sequence model's contribution scores and TF-MoDISco results, and
finding where the model used each motif. Scores are numpy arrays of shape
(sequences, letters, length) or (sequences, length). Reading HDF5 files needs
the `h5py` extra:

```sh
pip install "universalmotif[h5py]"
```

The [motifs and deep learning](../guides/deep-learning.ipynb) and
[TF-MoDISco results](../guides/tfmodisco.ipynb) guides use these functions on
a published model.

| Task | Function |
| --- | --- |
| Read contribution scores from chromBPNet, ENCODE or numpy files | [`read_contributions()`][universalmotif.read_contributions] |
| Read scores from a bigWig track | [`signal_values()`][universalmotif.signal_values] |
| Read TF-MoDISco patterns, or their seqlets | [`read_modisco()`][universalmotif.read_modisco], [`read_seqlets()`][universalmotif.read_seqlets] |
| Find where a model used each motif | [`scan_contributions()`][universalmotif.scan_contributions], with thresholds from [`modisco_thresholds()`][universalmotif.modisco_thresholds] |
| Gather the scores around motif hits | [`hit_contributions()`][universalmotif.hit_contributions] |
| Name patterns by a motif database | [`name_motifs()`][universalmotif.name_motifs] |
| Draw scores along a sequence | [`view_attributions()`][universalmotif.view_attributions], [`view_attribution_tracks()`][universalmotif.view_attribution_tracks] |
| Draw TF-MoDISco patterns and seqlets | [`plot_modisco()`][universalmotif.plot_modisco], [`plot_seqlets()`][universalmotif.plot_seqlets] |
| Draw the scores around hits | [`plot_hit_profile()`][universalmotif.plot_hit_profile], [`plot_hit_contributions()`][universalmotif.plot_hit_contributions] |

## Reading scores and TF-MoDISco results

::: universalmotif.read_contributions

::: universalmotif.ContributionScores
    options:
      members: [view]

::: universalmotif.read_modisco

::: universalmotif.read_seqlets

## Hits in contribution scores

::: universalmotif.scan_contributions

::: universalmotif.modisco_thresholds

::: universalmotif.hit_contributions

::: universalmotif.HitContributions
