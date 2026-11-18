# Build the report with Tectonic (brew install tectonic).
# Output goes to build/main.pdf (ignored by git).

.PHONY: pdf open clean zip

pdf:
	@mkdir -p build
	tectonic --keep-logs --outdir build main.tex

open: pdf
	open build/main.pdf

clean:
	rm -rf build

# Zip of the sources, ready for Overleaf (New Project -> Upload Project).
zip:
	@mkdir -p build
	rm -f build/overleaf.zip
	zip -r build/overleaf.zip . -x 'build/*' -x '.DS_Store'
