# Deep learning

Motifs as convolution filters, sequences as one-hot arrays, and TF-MoDISco
results and contribution scores, for working with sequence models. Arrays
are numpy arrays in the layouts PyTorch and most sequence models use, so
they can be passed to `torch.from_numpy()`, and PyTorch tensors can be
passed back as they are.
Reading TF-MoDISco results and chromBPNet contribution scores needs the
`h5py` extra:

```sh
pip install "universalmotif[h5py]"
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

## Finding motifs in contribution scores

[`contribution_motifs()`][universalmotif.contribution_motifs] finds the
motifs a model relies on in one call, from its contribution scores, by way
of the functions below, along with composite motifs from
[`gapped_motif_finder()`][universalmotif.gapped_motif_finder] and the hits
of every motif.
[`call_seqlets()`][universalmotif.call_seqlets] calls seqlets, the short
stretches of high contribution, in a model's scores, and their letters can
go to [`motif_finder()`][universalmotif.motif_finder] to find the motifs the
model relies on, with
[`refine_motifs()`][universalmotif.refine_motifs] and
[`select_motifs()`][universalmotif.select_motifs] then fitting the motifs
to the seqlets and keeping those that reach enough of them. Degenerate
sites, such as spaced palindromes, can be found from seqlets rewritten in
the letters the model prefers by
[`idealize_sequences()`][universalmotif.idealize_sequences], with the
motifs then refined on the real sequences, and
[`HitContributions.to_cwms()`][universalmotif.HitContributions.to_cwms]
gives the CWMs of motifs' hits, for leaving out motifs that are the same to
the model.
[`scan_contributions()`][universalmotif.scan_contributions] finds where a
model relied on each TF-MoDISco pattern, by matching CWMs against the
contribution scores of every input, with thresholds for each pattern from
its own seqlets by [`modisco_thresholds()`][universalmotif.modisco_thresholds].

::: universalmotif.contribution_motifs

::: universalmotif.ContributionMotifs

::: universalmotif.call_seqlets

::: universalmotif.idealize_sequences

::: universalmotif.scan_contributions

::: universalmotif.modisco_thresholds

## Sequences and filters

::: universalmotif.one_hot

::: universalmotif.from_one_hot

::: universalmotif.one_hot_batches

::: universalmotif.shuffled_references

::: universalmotif.to_filters

::: universalmotif.from_filters

## Probing a model

These make batches of sequences for a model of your own, without depending
on any deep-learning library. [`mutagenesis()`][universalmotif.mutagenesis]
makes every single-letter substitution of a sequence, and
[`mutagenesis_effects()`][universalmotif.mutagenesis_effects] turns the
model's predictions for them into an array that
[`view_attributions()`][universalmotif.view_attributions] can draw.
[`insert_sites()`][universalmotif.insert_sites] writes or inserts the same
sites into background sequences,
[`delete_region()`][universalmotif.delete_region] deletes a stretch of
each, and [`site_spacings()`][universalmotif.site_spacings]
places two sites at a range of distances apart.
[`view_variant()`][universalmotif.view_variant] compares the scores of a
sequence with those of a variant of it.
[`variant_batches()`][universalmotif.variant_batches] reads the reference
and alternative sequences around each variant of a VCF file, as
[`read_vcf()`][universalmotif.read_vcf] reads it, so that a model can
score them, and
[`variant_motif_effects()`][universalmotif.variant_motif_effects] finds
the motif sites the same variants change, to set beside the model's
effects in [`plot_variant_effects()`][universalmotif.plot_variant_effects].
[`pair_mutagenesis()`][universalmotif.pair_mutagenesis] adds every pair of
substitutions, and [`interaction_effects()`][universalmotif.interaction_effects]
turns the predictions for them into the epistasis between each pair of
positions, which [`plot_interactions()`][universalmotif.plot_interactions]
draws as a map.

::: universalmotif.mutagenesis

::: universalmotif.mutagenesis_effects

::: universalmotif.variant_batches

::: universalmotif.pair_mutagenesis

::: universalmotif.interaction_effects

::: universalmotif.insert_sites

::: universalmotif.delete_region

::: universalmotif.site_spacings
