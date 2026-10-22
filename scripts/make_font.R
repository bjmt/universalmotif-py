#!/usr/bin/env Rscript
#
# Export the letter outlines that universalmotif draws logos with
# (fontDFroboto, made with gglogo's createPolygons() from Roboto Medium) to
# python/universalmotif/data/roboto.json.
#
# Usage (from the root of this repository):
#   Rscript scripts/make_font.R [path/to/universalmotif]

args <- commandArgs(trailingOnly = TRUE)
um_path <- normalizePath(if (length(args)) args[1] else "~/universalmotif")
suppressMessages(pkgload::load_all(um_path, quiet = TRUE, export_all = TRUE))

f <- as.data.frame(fontDFroboto)
f$group <- as.character(f$group)
f <- f[order(f$group, f$order), ]
glyphs <- lapply(split(f, f$group), function(g) list(x = g$x, y = g$y))
glyphs <- glyphs[unique(as.character(fontDFroboto$group))]
writeLines(jsonlite::toJSON(glyphs, digits = I(17), auto_unbox = TRUE),
           file.path("python", "universalmotif", "data", "roboto.json"))
