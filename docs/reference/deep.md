# Deep learning

Motifs as convolution filters, sequences as one-hot arrays, and TF-MoDISco
results, for working with sequence models. Arrays are numpy arrays in the
layouts PyTorch and most sequence models use, so they can be passed to
`torch.from_numpy()`, and PyTorch tensors can be passed back as they are.
Reading TF-MoDISco results needs the `h5py` extra:

```sh
pip install universalmotif[h5py]
```

::: universalmotif.to_filters

::: universalmotif.from_filters

::: universalmotif.one_hot

::: universalmotif.from_one_hot

::: universalmotif.read_modisco
