# Model inputs and experiments

Making inputs for a sequence model and reading its outputs back, without
depending on any deep-learning library. Sequences become numpy arrays of
shape (sequences, letters, length), the layout PyTorch and most models use.
Arrays can be passed to `torch.from_numpy()`, and PyTorch tensors can be
passed back as they are. The
[experiments on a model](../guides/model-experiments.ipynb) guide shows them
in use.

| Task | Function |
| --- | --- |
| One-hot encode sequences, or decode them | [`one_hot()`][universalmotif.one_hot], [`from_one_hot()`][universalmotif.from_one_hot] |
| Read genomic regions in batches for training | [`one_hot_batches()`][universalmotif.one_hot_batches], with backgrounds from [`match_bkg()`][universalmotif.match_bkg] |
| Make references for DeepLIFT and DeepSHAP | [`shuffled_references()`][universalmotif.shuffled_references] |
| Turn motifs into convolution filters, or back | [`to_filters()`][universalmotif.to_filters], [`from_filters()`][universalmotif.from_filters] |
| Change every base of a sequence | [`mutagenesis()`][universalmotif.mutagenesis], [`mutagenesis_effects()`][universalmotif.mutagenesis_effects] |
| Change every pair of bases | [`pair_mutagenesis()`][universalmotif.pair_mutagenesis], [`interaction_effects()`][universalmotif.interaction_effects], [`plot_interactions()`][universalmotif.plot_interactions] |
| Insert sites, or place two sites at a range of spacings | [`insert_sites()`][universalmotif.insert_sites], [`site_spacings()`][universalmotif.site_spacings], [`plot_marginalization()`][universalmotif.plot_marginalization], [`plot_spacing()`][universalmotif.plot_spacing] |
| Delete or shuffle a stretch of each sequence | [`delete_region()`][universalmotif.delete_region], [`shuffle_region()`][universalmotif.shuffle_region] |
| Read variants for a model | [`variant_batches()`][universalmotif.variant_batches], with [`read_vcf()`][universalmotif.read_vcf] |
| Compare variants with the motif sites they change | [`variant_motif_effects()`][universalmotif.variant_motif_effects], [`plot_variant_effects()`][universalmotif.plot_variant_effects], [`view_variant()`][universalmotif.view_variant] |

## Sequences and filters

::: universalmotif.one_hot

::: universalmotif.from_one_hot

::: universalmotif.one_hot_batches

::: universalmotif.shuffled_references

::: universalmotif.to_filters

::: universalmotif.from_filters

## Experiments

::: universalmotif.mutagenesis

::: universalmotif.mutagenesis_effects

::: universalmotif.pair_mutagenesis

::: universalmotif.interaction_effects

::: universalmotif.insert_sites

::: universalmotif.delete_region

::: universalmotif.site_spacings

## Variants

::: universalmotif.variant_batches
