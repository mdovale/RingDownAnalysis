MAIN := slides
NOTES_MAIN := slides-notes
PDFLATEX_FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error

# Prefer the repo virtualenv for figure regeneration when present.
VENV_PY := ../../../.venv/bin/python
PY := $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)

.PHONY: pdf pdf-notes figures cache slides-png clean

pdf:
	latexmk -pdf $(PDFLATEX_FLAGS) $(MAIN).tex

# Printable / rehearsal PDF: each page = slide + speaker notes below.
pdf-notes:
	latexmk -pdf $(PDFLATEX_FLAGS) $(NOTES_MAIN).tex

# Cheap: reads assets/_cache only, no access to the raw phasemeter exports.
figures:
	$(PY) assets/make_figures.py

# Expensive: re-reads the multi-GB records and rebuilds assets/_cache.
cache:
	$(PY) assets/build_cache.py

# Static-render each page for visual review (ImageMagick; pdftoppm not present here).
slides-png: pdf
	mkdir -p _review
	magick -density 150 $(MAIN).pdf -background white -alpha remove -alpha off _review/slide-%02d.png

clean:
	latexmk -c $(MAIN).tex
	latexmk -c $(NOTES_MAIN).tex
	rm -rf _review assets/_preview
