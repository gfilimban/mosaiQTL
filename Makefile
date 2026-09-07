# mosaiQTL Makefile
# Install: pip install -e '.[dev]'

.PHONY: lint test-unit test coverage \
        validate-phase1 validate-phase1-p2 \
        validate-phase2 validate-section validate-force \
        validate-phase3 validate-phase3-quick validate-phase3-force \
        validate-phase4 \
        validate-tissue validate-tissue-scan validate-tissue-full \
        validate-tissue-plots validate-tissue-compare \
        validate-tissue-all-cts \
        clean clean-phase3

# ── lint ──────────────────────────────────────────────────────────────────────
lint:
	black --check src/ tests/
	flake8 src/ tests/
	interrogate --fail-under 100 src/mosaiqtl/
	mypy --strict src/mosaiqtl/

# ── tests ─────────────────────────────────────────────────────────────────────
test-unit:
	pytest tests/unit/ -x -q

test:
	pytest tests/unit/ tests/integration/ --ignore=tests/benchmarks/ -q

coverage:
	pytest tests/unit/ tests/integration/ --ignore=tests/benchmarks/ \
	    --cov=src/mosaiqtl --cov-report=term-missing -q

# ── Phase 1 — minimal miQTL workflow mirror ───────────────────────────────────
# Run R first: Rscript validation/r_reference/port_sim_validation.R
validate-phase1:
	python validation/validate_sim.py

# Part 2: sparcc_cache + sim.CC.data
# Run R first: Rscript validation/r_reference/port_sim_validation_p2.R
validate-phase1-p2:
	python validation/validate_sim_p2.py

# ── Phase 2 — full miQTL port (S1–S11) ───────────────────────────────────────
validate-phase2:
	python validation/validate_port.py

validate-section:
	@echo "usage: make validate-section SECTIONS=2,11"
	python validation/validate_port.py --sections $(SECTIONS)

validate-force:
	python validation/validate_port.py --force

# ── Phase 3 — speed + memory ──────────────────────────────────────────────────
validate-phase3:
	python validation/validate_phase3.py --cell_type type2x --n_genes 500

validate-phase3-quick:
	python validation/validate_phase3.py --cell_type type2x --n_genes 100 \
	    --sections nb,vqtl

validate-phase3-force:
	python validation/validate_phase3.py --cell_type type2x --n_genes 500 --force

# ── Phase 4 — NB-GLMM vs LMM ─────────────────────────────────────────────────
validate-phase4:
	python validation/validate_phase4.py --cell_type type2x --n_genes 500

# ── Tissue validation (default: gastroc) ─────────────────────────────────────
# Prereq: python validation/tools/convert_h5ad.py
# Then R: Rscript validation/r_reference/gastroc_miqtl_scan.R all 8 C
validate-tissue:
	python validation/validate_tissue.py

validate-tissue-scan:
	python validation/validate_tissue.py --scan

validate-tissue-full:
	python validation/validate_tissue.py --full

validate-tissue-plots:
	python validation/validate_tissue.py --plots-only

validate-tissue-compare:
	python validation/validate_tissue.py --compare

validate-tissue-all-cts:
	python validation/validate_tissue.py --all-celltypes

# ── clean ─────────────────────────────────────────────────────────────────────
clean:
	find validation/phase2 -name '*.png' -o -name '*.csv' -o -name '*.done' 2>/dev/null \
	    | xargs rm -f
	@echo "cleared phase2 outputs"

clean-phase3:
	find validation/phase3 -name '*.png' -o -name '*.csv' -o -name '*.done' 2>/dev/null \
	    | xargs rm -f
	@echo "cleared phase3 outputs"

# ── aliases ───────────────────────────────────────────────────────────────────
validate: validate-phase1
