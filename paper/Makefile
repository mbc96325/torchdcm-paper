PDF = main.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): main.tex references.bib sections/*.tex tables/*.tex
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc $(PDF)
