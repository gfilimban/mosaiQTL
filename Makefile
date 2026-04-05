# mosaiQTL Makefile
# ─────────────────────────────────────────────────────────────────────────────
# bundle workflow:
#   sender:   make bundle           (creates mosaiqtl.bundle)
#   receiver: git clone mosaiqtl.bundle mosaiqtl --branch main  (first time)
#             git pull /path/to/mosaiqtl.bundle main --rebase   (updates)
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: bundle check test validate validate-chr8 validate-phase2 \
        validate-phase3 validate-phase3-quick clean clean-phase3

# ── bundle ───────────────────────────────────────────────────────────────────
bundle:
	git bundle create mosaiqtl.bundle main
	@echo "created mosaiqtl.bundle  ($(shell git log -1 --format='%h %s'))"
	@echo "pull:  git pull /path/to/mosaiqtl.bundle main --rebase"

# ── syntax check ─────────────────────────────────────────────────────────────
check:
	python -m py_compile mosaiqtl/*.py validation/validate_port.py validation/validate_phase3.py
	@echo "syntax OK"

# ── unit tests ───────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

# ── phase 2 validation (S1–S11, port vs miQTL R) ─────────────────────────────
validate-phase2:
	python validation/validate_port.py

validate-chr8:
	python validation/validate_port.py --chr 8 --n_genes 50

validate-section:
	@echo "usage: make validate-section SECTIONS=2,11"
	python validation/validate_port.py --sections $(SECTIONS)

# ── phase 3 validation (HPC — requires gastroc_all_processed2.h5ad) ──────────
validate-phase3:
	python validation/validate_phase3.py --cell_type type2x --n_genes 500

validate-phase3-quick:
	python validation/validate_phase3.py --cell_type type2x --n_genes 100 \
	    --sections nb,vqtl

# ── force rerun ignoring checkpoints ─────────────────────────────────────────
validate-force:
	python validation/validate_port.py --force

validate-phase3-force:
	python validation/validate_phase3.py --cell_type type2x --n_genes 500 --force

# validate native NEBULA-LN vs edgepython (P1=local; P2=HPC with --h5ad)
validate-nebula-ln:
	python validation/validate_nebula_ln.py

validate-nebula-ln-hpc:
	python validation/validate_nebula_ln.py --h5ad /path/to/gastroc_all_processed2.h5ad \
	    --cell_type type2x --n_genes 500

# ── clean ─────────────────────────────────────────────────────────────────────
clean:
	find validation/phase2 -name '*.png' -o -name '*.csv' -o -name '*.done' | xargs rm -f
	@echo "cleared phase2 outputs"

clean-phase3:
	find validation/phase3 -name '*.png' -o -name '*.csv' -o -name '*.done' | xargs rm -f
	@echo "cleared phase3 outputs"

# alias
validate: validate-phase2
