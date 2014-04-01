LATEX = latex
MAIN  = ms

$(MAIN).pdf: 	$(MAIN).tex
	$(LATEX) $(MAIN).tex 
#	bibtex $(MAIN)
	$(LATEX) $(MAIN).tex 
	$(LATEX) $(MAIN).tex 
	dvips $(MAIN).dvi -o $(MAIN).ps
	ps2pdf $(MAIN).ps 

diff:
	latexdiff $(MAIN)_old.tex $(MAIN).tex > $(MAIN)_diff.tex
	$(LATEX) $(MAIN)_diff.tex 
	bibtex $(MAIN)_diff
	$(LATEX) $(MAIN)_diff.tex 
	$(LATEX) $(MAIN)_diff.tex 
	dvips $(MAIN)_diff.dvi -o $(MAIN)_diff.ps
	ps2pdf $(MAIN)_diff.ps 

clean:
	rm -f $(MAIN).aux
	rm -f $(MAIN).bbl
	rm -f $(MAIN).ps
	rm -f $(MAIN).pdf
	rm -f $(MAIN).blg
	rm -f $(MAIN).log
	rm -f $(MAIN).dvi
	rm -f *~
view: 
	open $(MAIN).pdf &

