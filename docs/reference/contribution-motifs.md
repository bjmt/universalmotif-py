# Motifs from contribution scores

Finding the motifs a model relies on from its contribution scores, without
TF-MoDISco. [`contribution_motifs()`][universalmotif.contribution_motifs]
does every step in one call. Each step is also a function of its own, for
more control. The [motifs from contribution scores](../guides/contribution-motifs.ipynb)
guide works through them on a published model.

| Step | Function |
| --- | --- |
| Call seqlets, the stretches of high contribution | [`call_seqlets()`][universalmotif.call_seqlets], with [`seqlet_fdr()`][universalmotif.seqlet_fdr] for seqlets called in batches |
| Rewrite seqlets in the letters the model prefers | [`idealize_sequences()`][universalmotif.idealize_sequences] |
| Search the seqlets for motifs | [`motif_finder()`][universalmotif.motif_finder], [`gapped_motif_finder()`][universalmotif.gapped_motif_finder] |
| Rebuild motifs from their sites in the seqlets | [`refine_motifs()`][universalmotif.refine_motifs] |
| Keep the motifs that reach the most seqlets | [`select_motifs()`][universalmotif.select_motifs] |
| Give motifs as CWMs from their hits | [`HitContributions.to_cwms()`][universalmotif.HitContributions.to_cwms] |
| Find where the model used each motif | [`scan_contributions()`][universalmotif.scan_contributions] |
| Name the motifs by a motif database | [`name_motifs()`][universalmotif.name_motifs] |
| Gather the patterns of many runs into one set | [`combine_patterns()`][universalmotif.combine_patterns] |

::: universalmotif.contribution_motifs

::: universalmotif.ContributionMotifs

::: universalmotif.call_seqlets

::: universalmotif.seqlet_fdr

::: universalmotif.idealize_sequences

## Patterns of many runs

::: universalmotif.combine_patterns

::: universalmotif.CombinedPatterns
