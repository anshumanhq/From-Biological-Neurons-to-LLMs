# ============================================================
# MAKEFILE for From Biological Neurons to LLMs
# ============================================================

SHELL = /bin/bash
NAME = main

.PHONY: all clean pdf watch

all: pdf

pdf:
	latexmk -pdf $(NAME)

watch:
	latexmk -pvc -pdf $(NAME)

clean:
	latexmk -C
	rm -f *.bbl *.blg *.run.xml *.aux *.log *.out *.toc *.lof *.lot *.bcf