#!/bin/bash
# Quick script to rebuild references after modifying .bib file

echo "🔄 Cleaning old files..."
rm -f *.aux *.bbl *.blg *.log *.out

echo "📚 Building with references..."
pdflatex -interaction=nonstopmode IEEE-conference-template-062824.tex > /dev/null
bibtex IEEE-conference-template-062824
pdflatex -interaction=nonstopmode IEEE-conference-template-062824.tex > /dev/null
pdflatex -interaction=nonstopmode IEEE-conference-template-062824.tex > /dev/null

echo "✅ Done! Check IEEE-conference-template-062824.pdf"

