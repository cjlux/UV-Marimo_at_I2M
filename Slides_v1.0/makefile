all:
	pdflatex -shell-escape slides.tex && biber slides && pdflatex -shell-escape slides && pdflatex -shell-escape slides && evince slides.pdf&

simple:
	pdflatex  -shell-escape slides.tex && evince slides.pdf&

clean:
	rm -f  *.sov *.vrb *.aux *.toc *.log *.nav *.out *.snm *.blg *.bbl *.bcf *.run.xml *.brf *~ renum.tex

renum:
	cp slides.tex slides.tex.sov
	python3 renumber-slides.py && mv renum.tex slides.tex
