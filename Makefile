# mosaiQTL Makefile
# ─────────────────────────────────────────────────────────────────────────────
# bundle: create mosaiqtl.bundle for transfer to HPC or collaborators.
#   always named mosaiqtl.bundle so the pull command never changes.
#
# receiver workflow (first time):
#   git clone mosaiqtl.bundle mosaiqtl --branch main
#
# receiver workflow (updates):
#   git pull /absolute/path/to/mosaiqtl.bundle main
#   — or, if bundle is in the repo parent dir:
#   git pull ../mosaiqtl.bundle main
#
# sender workflow (after any commit):
#   make bundle
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: bundle validate clean

bundle:
	git bundle create mosaiqtl.bundle main
	@echo "created mosaiqtl.bundle  ($(shell git log -1 --format='%h %s'))"
	@echo "pull with:  git pull /path/to/mosaiqtl.bundle main"

# run genome-wide validation (HPC) — adjust chr/n_genes as needed
validate:
	python validation/validate_port.py

# run single chromosome for quick checks
validate-chr8:
	python validation/validate_port.py --chr 8 --n_genes 50

clean:
	find validation/port_validation -name '*.png' -delete
	find validation/port_validation -name '*.csv' -delete
	@echo "cleared qc_plots/ and py_data/"

# phase 3 validation (HPC only — requires gastroc_all_processed2.h5ad)
validate-phase3:
	python validation/validate_phase3.py --cell_type type2x --n_genes 500

validate-phase3-quick:
	python validation/validate_phase3.py --cell_type type2x --n_genes 100 \
	    --sections nb,vqtl
