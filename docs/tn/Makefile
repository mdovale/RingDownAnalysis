# Makefile for technical note generation
# Handles plot generation, LaTeX compilation, and cleanup

# Python script to generate figures
FIGURES_SCRIPT = generate_figures.py

# LaTeX document
LATEX_DOC = main.tex
PDF_OUTPUT = main.pdf

# Python executable (use python3 if python is not available)
PYTHON = python3

# LaTeX compiler
LATEX = pdflatex

# LaTeX compilation flags
LATEX_FLAGS = -interaction=nonstopmode -file-line-error

# Generated figure files (PDFs)
FIGURES = \
	freq_estimation_ringdown_v6_individual.pdf \
	freq_estimation_ringdown_v6_aggregate.pdf \
	freq_estimation_ringdown_v6_performance.pdf \
	q_estimation_ringdown_v6_individual.pdf \
	q_estimation_ringdown_v6_performance.pdf \
	crlb_freq_vs_tau_ratio.pdf \
	crlb_freq_vs_snr.pdf \
	crlb_q_vs_q.pdf \
	crlb_q_vs_tau_ratio.pdf

# LaTeX auxiliary files to clean
LATEX_AUX = \
	*.aux \
	*.log \
	*.out \
	*.toc \
	*.bbl \
	*.blg \
	*.synctex.gz \
	*.fls \
	*.fdb_latexmk

.PHONY: all figures latex clean help

# Default target: generate everything
all: figures latex

# Generate all figures
figures:
	@echo "Generating figures..."
	$(PYTHON) $(FIGURES_SCRIPT)
	@echo "Figures generated successfully."

# Compile LaTeX document (run twice for cross-references)
latex: figures
	@echo "Compiling LaTeX document (first pass)..."
	$(LATEX) $(LATEX_FLAGS) $(LATEX_DOC) > /dev/null || true
	@echo "Compiling LaTeX document (second pass for cross-references)..."
	$(LATEX) $(LATEX_FLAGS) $(LATEX_DOC) > /dev/null || true
	@echo "LaTeX compilation completed."

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f $(FIGURES)
	rm -f $(LATEX_AUX)
	rm -f $(PDF_OUTPUT)
	@echo "Cleanup completed."

# Show help message
help:
	@echo "Makefile for technical note generation"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Generate figures and compile LaTeX document (default)"
	@echo "  figures  - Generate all figure PDFs"
	@echo "  latex    - Compile LaTeX document (requires figures)"
	@echo "  clean    - Remove all generated files (figures, PDF, LaTeX aux files)"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Generate everything"
	@echo "  make figures      # Generate only figures"
	@echo "  make latex        # Compile LaTeX (regenerates figures if needed)"
	@echo "  make clean        # Remove all generated files"
