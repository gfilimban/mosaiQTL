# docs/HPC.md — HPC Setup & Cluster Operations
# UCI CRSP cluster (SLURM). Never auto-loaded by Claude Code.

---

## PATHS (never commit these — set manually per script)

```
cluster:     hpc3.rcic.uci.edu
account:     seyedam
scratch:     /tmp/$USER/$SLURM_JOB_ID   (node-local; rsync to results/ on success)
pkg_dir:     /share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl
miqtl_dir:  /share/crsp/lab/seyedam/gfilimba/projects/igvf_8cube/short-read/data/gastroc
adata:       {miqtl_dir}/gastroc_all_processed2.h5ad
r_data:      {pkg_dir}/validation/port_validation/r_data
py_data:     {pkg_dir}/validation/port_validation/py_data
conda_env:   bioinfo  (Python 3.13)
containers:  /share/crsp/lab/seyedam/gfilimba/containers/
```

---

## GIT BUNDLE WORKFLOW (no internet on compute nodes)

```bash
# local machine
make bundle                           # creates mosaiqtl.bundle
scp mosaiqtl.bundle hpc3:/tmp/

# on HPC
cd /share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl
git config user.email 'gfilimba@uci.edu'
git config user.name  'gfilimba'
git pull /tmp/mosaiqtl.bundle main --rebase
git log --oneline -3                  # verify commit
pip install -e '.'                    # reinstall after changes
```

---

## SLURM BASICS

```bash
squeue -u gfilimba                    # check running jobs
scancel <jobid>                       # cancel job
seff <jobid>                          # efficiency report after completion
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # resource usage
```

---

## NEXTFLOW SUBMISSION

```bash
# set account (required — never commit)
export SLURM_ACCOUNT=seyedam

# dry-run (validates DAG without cluster submission)
nextflow run main.nf -profile local -stub-run \
    -params-file config/default_params.yaml

# submit to cluster
nextflow run main.nf -profile slurm \
    -params-file config/default_params.yaml

# resume after failure (idempotent — re-uses cached work/)
nextflow run main.nf -profile slurm -resume \
    -params-file config/default_params.yaml

# with reports
nextflow run main.nf -profile slurm \
    -with-report results/reports/run_report.html \
    -with-timeline results/reports/timeline.html \
    -with-trace results/reports/trace.txt \
    -params-file config/default_params.yaml

# clean work directory after successful run
nextflow clean -f
```

---

## SINGULARITY / APPTAINER

```bash
# build (requires Docker on a machine with internet)
docker build -t mosaiqtl:0.2.0 .
singularity build mosaiqtl_0.2.0.sif docker-daemon://mosaiqtl:0.2.0

# transfer to HPC
scp mosaiqtl_0.2.0.sif hpc3:/share/crsp/lab/seyedam/gfilimba/containers/

# set in nextflow.config
# container = '/share/crsp/lab/seyedam/gfilimba/containers/mosaiqtl_0.2.0.sif'
# containerOptions = '--bind /share/crsp/'
```

---

## VALIDATION ON HPC

```bash
conda activate bioinfo
cd /share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl

# Phase 2 — R then Python
Rscript validation/r_reference/port_validation.R all Inf 2
python validation/validate_port.py
python validation/validate_port.py --sections 2,6,11

# Phase 3 — biological consistency (requires adata)
python validation/validate_phase3.py --cell_type type2x --n_genes 500
make validate-phase3-quick   # n_genes=100, sections nb,vqtl
```

---

## REFERENCES
- Di Tommaso P et al. Nextflow. *Nat Biotechnol* 2017. DOI: 10.1038/nbt.3820
- Ewels PA et al. nf-core. *Nat Biotechnol* 2020. DOI: 10.1038/s41587-020-0439-x
- Kurtzer GM et al. Singularity. *PLOS ONE* 2017. DOI: 10.1371/journal.pone.0177459
