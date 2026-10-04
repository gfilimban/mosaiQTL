"""
config — Population, ScanConfig, and PipelineConfig for mosaiQTL

Population is the first-class runtime type for multiparent population configuration.
ScanConfig holds per-scan parameters; PipelineConfig orchestrates multi-scan runs.

import note: this module must NOT import from genotype/ or qtl/ — those modules
import Population from here, so the dependency must be one-directional.
ScanModel / HaplotypeModel / TestStatistic are re-declared here (not imported from
records.py) to keep config.py importable without the full mosaiqtl package installed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ScanModel: type = Literal["nb-glmm", "lmm"]
HaplotypeModel: type = Literal["additive", "full", "mi"]
TestStatistic: type = Literal["score", "lrt", "f_test"]

_MM39_AUTOSOMES: list[str] = [f"chr{i}" for i in range(1, 20)]
_MM39_CHROMS: list[str] = _MM39_AUTOSOMES + ["chrX"]


class Population(BaseModel, frozen=True):
    """Multiparent population configuration.

    frozen=True: instances are hashable and safe to use as dict keys or
    cache keys. Pass a Population instance wherever population identity is
    needed; pass .name (str) for serialisation.

    attributes
    ----------
    name : str
        short identifier, e.g. 'cc', 'do', 'magic19'
    founders : list[str]
        ordered founder strain names — defines haplotype axis order in all
        arrays and plots. never infer order from a file; always use this list.
    colors : dict[str, str]
        founder name → hex color string; must contain an entry for every
        founder in founders (validated)
    f1_colors : dict[str, str]
        F1 hybrid name → hex color; optional, default empty
    chromosome_names : list[str]
        ordered chromosome identifiers in UCSC style ('chrN'); default mm39
    has_sex_chromosomes : bool
        True for typical diploid organisms with X/Y; False for polyploid or
        asexual designs where sex-chromosome QTL handling is not needed
    sex_chromosome : str
        primary sex chromosome identifier (e.g. 'chrX')
    x_donors : list[str]
        subset of founders that donated the X chromosome; required for
        correct sex-chromosome dosage compensation and QTL handling
    genotyping_error_prob : float
        per-locus genotyping error probability for the HMM forward-backward
        algorithm; 0.002 is the standard default for GigaMUGA CC data
    n_gen : int
        inbreeding generations used by the HMM transition matrix;
        CC = 1 (consensus map), DO = animal-specific (varies)
    inbred : bool
        True for fully inbred lines (CC); False for outbred (DO, F2)
    loco_sufficient : bool
        True when LOCO kinship alone fully accounts for population structure
        (CC, DO, F2, MPP); False for structured populations where genotype
        PCs are also needed alongside LOCO kinship
    """

    name: str
    founders: list[str]
    colors: dict[str, str]
    f1_colors: dict[str, str] = Field(default_factory=dict)
    chromosome_names: list[str] = Field(default_factory=lambda: list(_MM39_CHROMS))
    has_sex_chromosomes: bool = True
    sex_chromosome: str = "chrX"
    x_donors: list[str] = Field(default_factory=list)
    genotyping_error_prob: float = 0.002
    n_gen: int = 1
    inbred: bool = True
    loco_sufficient: bool = True

    @property
    def n_founders(self) -> int:
        """Number of founder strains."""
        return len(self.founders)

    @model_validator(mode="after")
    def _colors_cover_founders(self) -> "Population":
        """Validate that colors dict contains an entry for every founder."""
        missing = [f for f in self.founders if f not in self.colors]
        if missing:
            raise ValueError(
                f"Population.colors missing entries for founders: {missing}"
            )
        return self


class ScanConfig(BaseModel):
    """Per-scan configuration (Tier 2 axes — exact reproduction parameters).

    attributes
    ----------
    population : str
        population identifier; use 'cc' for Collaborative Cross.
        pass a Population.name — resolved to a Population instance at runtime
        via a population registry (genotype/founders.py for built-ins)
    model : str
        statistical model: 'nb-glmm' (primary) or 'lmm' (validated port)
    test_statistic : str
        test statistic: 'score' (nb-glmm default), 'lrt', or 'f_test' (lmm)
    haplotype : str
        haplotype encoding: 'additive', 'full', or 'mi' (multiple imputation)
    scan_window : str
        region scanned: 'cis', 'chr', 'genome', or 'loci=<comma-list>'
    offset : str
        library size offset: 'from_libsize', 'from_file', or 'none'
    n_perms : int
        number of permutations for GEV threshold estimation; minimum 100
    seed : int
        random seed for permutations and MI draws
    cis_window_mb : float
        cis window half-width in Mb for locus_type assignment
    permutation_method : str
        'plain' (Phase 1 — unconditional shuffle) or 'parametric'
        (Phase 2+ — kinship-informed rank-permutation preserving genetic structure)
    """

    population: str = "cc"
    model: str = "nb-glmm"
    test_statistic: str = "score"
    haplotype: str = "additive"
    scan_window: str = "chr"
    offset: str = "from_libsize"
    n_perms: int = Field(default=1000, ge=100)
    seed: int = 42
    cis_window_mb: float = Field(default=10.0, gt=0.0)
    permutation_method: str = "plain"


class PipelineConfig(BaseModel):
    """Pipeline orchestration configuration for multi-scan runs.

    attributes
    ----------
    scan_config : ScanConfig
        per-scan parameters applied uniformly to all cell types
    input_h5 : str
        path to pseudobulk count HDF5 (output of user upstream pipeline)
    genome_cache_h5 : str
        path to HDF5 genome cache in mosaiQTL native format
    output_dir : str
        directory where scan CSVs and QC plots are written
    cell_types : list[str]
        cell type labels to include; scans all available if empty
    n_jobs : int
        number of parallel scan workers (per-chromosome parallelism); >= 1
    """

    scan_config: ScanConfig
    input_h5: str
    genome_cache_h5: str
    output_dir: str
    cell_types: list[str] = Field(default_factory=list)
    n_jobs: int = Field(default=1, ge=1)
