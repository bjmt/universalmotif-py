# Deep learning

Motifs as convolution filters, sequences as one-hot arrays, and TF-MoDISco
results and contribution scores, for working with sequence models. Arrays
are numpy arrays in the layouts PyTorch and most sequence models use, so
they can be passed to `torch.from_numpy()`, and PyTorch tensors can be
passed back as they are.
Reading TF-MoDISco results and chromBPNet contribution scores needs the
`h5py` extra:

```sh
pip install universalmotif[h5py]
```

Attribution and contribution scores from a model, for each position or for
each letter at each position, can be drawn along their sequence with
[`view_attributions()`][universalmotif.view_attributions]. Scores saved by
chromBPNet or in the numpy files TF-MoDISco takes are read with
[`read_contributions()`][universalmotif.read_contributions], and scores kept
as bigWig tracks with [`signal_values()`][universalmotif.signal_values].
The [motifs and deep learning](../guides/deep-learning.ipynb) guide puts
these together on the outputs of a published model.

## Reading and drawing scores

::: universalmotif.read_modisco

::: universalmotif.read_seqlets

::: universalmotif.read_contributions

::: universalmotif.ContributionScores
    options:
      members: [view]

::: universalmotif.hit_contributions

::: universalmotif.HitContributions

## Sequences and filters

::: universalmotif.one_hot

::: universalmotif.from_one_hot

::: universalmotif.one_hot_batches

::: universalmotif.to_filters

::: universalmotif.from_filters

## Probing a model

These make batches of sequences for a model of your own, without depending
on any deep-learning library. [`mutagenesis()`][universalmotif.mutagenesis]
makes every single-letter substitution of a sequence, and
[`mutagenesis_effects()`][universalmotif.mutagenesis_effects] turns the
model's predictions for them into an array that
[`view_attributions()`][universalmotif.view_attributions] can draw.
[`insert_sites()`][universalmotif.insert_sites] writes the same sites into
background sequences, and [`site_spacings()`][universalmotif.site_spacings]
places two sites at a range of distances apart.
[`view_variant()`][universalmotif.view_variant] compares the scores of a
sequence with those of a variant of it.

::: universalmotif.mutagenesis

::: universalmotif.mutagenesis_effects

::: universalmotif.insert_sites

::: universalmotif.site_spacings
