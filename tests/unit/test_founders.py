# tests/unit/test_founders.py — unit tests for genotype/founders.py
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from mosaiqtl.genotype.founders import (
    FOUNDER_COLORS,
    FOUNDER_ORDER,
    F1_COLORS,
    STRAIN_ORDER,
    cc_strain_palette,
)


class TestFounderOrder:
    def test_founder_order_length(self):
        assert len(FOUNDER_ORDER) == 8

    def test_founder_order_is_list(self):
        assert isinstance(FOUNDER_ORDER, list)

    def test_founder_colors_keys_match_order(self):
        assert set(FOUNDER_COLORS.keys()) == set(FOUNDER_ORDER)

    def test_founder_colors_valid_hex(self):
        pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for name, color in FOUNDER_COLORS.items():
            assert pattern.match(color), f"{name}: {color!r} is not valid hex"


class TestF1Colors:
    def test_f1_colors_complete(self):
        assert len(F1_COLORS) == 7

    def test_f1_colors_valid_hex(self):
        pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for name, color in F1_COLORS.items():
            assert pattern.match(color), f"{name}: {color!r} is not valid hex"


class TestStrainOrder:
    def test_strain_order_founders_first(self):
        assert STRAIN_ORDER[:8] == FOUNDER_ORDER

    def test_strain_order_f1s_after_founders(self):
        assert STRAIN_ORDER[8:15] == list(F1_COLORS.keys())


class TestCcStrainPalette:
    def test_cc_palette_returns_dict(self):
        result = cc_strain_palette(["CC001", "CC002"])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"CC001", "CC002"}

    def test_cc_palette_no_probs_returns_grey(self):
        result = cc_strain_palette(["CC001"])
        assert result["CC001"] == "#888888"

    def test_cc_palette_with_probs_valid_hex(self):
        # A/J-dominant strain
        probs = {"CC001": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        result = cc_strain_palette(["CC001"], haplotype_probs=probs)
        pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        assert pattern.match(result["CC001"])

    def test_cc_palette_pure_founder_matches_founder_color(self):
        # strain with 100% A/J ancestry → should equal A/J color
        probs = {"CC001": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        result = cc_strain_palette(["CC001"], haplotype_probs=probs)
        assert result["CC001"].upper() == FOUNDER_COLORS["AJ"].upper()

    def test_cc_palette_sorted_keys(self):
        strains = ["CC010", "CC002", "CC005"]
        result = cc_strain_palette(strains)
        assert list(result.keys()) == sorted(strains)


class TestNoInlineHexElsewhere:
    def test_no_inline_hex_outside_founders(self):
        src = Path(__file__).parents[2] / "src" / "mosaiqtl"
        hex_re = re.compile(r"'#[0-9A-Fa-f]{6}'|\"#[0-9A-Fa-f]{6}\"")
        # compare.py: R/Python series colors (#D83026, #4F6EAF).
        # _theme.py: package identity palette — canonical Layer-1 constants.
        _approved = {"compare.py", "_theme.py"}
        violations = []
        for py_file in src.rglob("*.py"):
            if py_file.name == "founders.py":
                continue
            if py_file.name in _approved:
                continue
            text = py_file.read_text()
            if hex_re.search(text):
                violations.append(str(py_file.relative_to(src.parent.parent)))
        assert (
            not violations
        ), "hardcoded hex colors found outside founders.py:\n" + "\n".join(violations)
