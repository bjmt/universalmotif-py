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

::: universalmotif.to_filters

::: universalmotif.from_filters

::: universalmotif.one_hot

::: universalmotif.from_one_hot

::: universalmotif.read_modisco

::: universalmotif.read_contributions

::: universalmotif.ContributionScores
    options:
      members: [view]
