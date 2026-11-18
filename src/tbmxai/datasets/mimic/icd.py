"""ICD-9-CM / ICD-10-CM helpers (MIMIC codes have no dots)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

#: ICD-9-CM numeric chapter ranges (inclusive) → shared chapter names.
_ICD9_CHAPTERS: list[tuple[int, int, str]] = [
    (1, 139, "infectious"),
    (140, 239, "neoplasms"),
    (240, 279, "endocrine"),
    (280, 289, "blood"),
    (290, 319, "mental"),
    (320, 389, "nervous"),
    (390, 459, "circulatory"),
    (460, 519, "respiratory"),
    (520, 579, "digestive"),
    (580, 629, "genitourinary"),
    (630, 679, "pregnancy"),
    (680, 709, "skin"),
    (710, 739, "musculoskeletal"),
    (740, 759, "congenital"),
    (760, 779, "perinatal"),
    (780, 799, "symptoms"),
    (800, 999, "injury"),
]

#: ICD-10-CM first letter → shared chapter names (D and H are split below).
_ICD10_LETTERS: dict[str, str] = {
    "A": "infectious",
    "B": "infectious",
    "C": "neoplasms",
    "E": "endocrine",
    "F": "mental",
    "G": "nervous",
    "I": "circulatory",
    "J": "respiratory",
    "K": "digestive",
    "L": "skin",
    "M": "musculoskeletal",
    "N": "genitourinary",
    "O": "pregnancy",
    "P": "perinatal",
    "Q": "congenital",
    "R": "symptoms",
    "S": "injury",
    "T": "injury",
    "V": "external",
    "W": "external",
    "X": "external",
    "Y": "external",
    "Z": "factors",
    "U": "special",
}

CHAPTERS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [c for *_, c in _ICD9_CHAPTERS] + list(_ICD10_LETTERS.values()) + ["external", "factors"]
    )
)


def normalize(code: object) -> str:
    """Upper-case, strip, remove dots."""
    return str(code).strip().upper().replace(".", "")


def chapter(code: str, version: int) -> str:
    """Shared chapter name of an ICD-9 or ICD-10 code (``"unknown"`` if unparseable)."""
    code = normalize(code)
    if not code:
        return "unknown"
    if int(version) == 10:
        letter = code[0]
        if letter == "D":
            return "neoplasms" if code[1:3] < "50" else "blood"
        if letter == "H":
            return "nervous"  # eye/ear grouped with nervous system as in ICD-9 320–389
        return _ICD10_LETTERS.get(letter, "unknown")
    if code[0] == "V":
        return "factors"
    if code[0] == "E":
        return "external"
    try:
        num = int(code[:3])
    except ValueError:
        return "unknown"
    return next((name for lo, hi, name in _ICD9_CHAPTERS if lo <= num <= hi), "unknown")


def match_prefixes(
    codes: pd.Series, versions: pd.Series, icd9: Iterable[str], icd10: Iterable[str]
) -> np.ndarray:
    """Boolean mask: code starts with any version-appropriate prefix."""
    codes = codes.map(normalize)
    versions = versions.astype(int)
    p9, p10 = tuple(map(normalize, icd9)), tuple(map(normalize, icd10))
    hit9 = (versions == 9) & codes.str.startswith(p9) if p9 else pd.Series(False, index=codes.index)
    hit10 = (
        (versions == 10) & codes.str.startswith(p10) if p10 else pd.Series(False, index=codes.index)
    )
    return (hit9 | hit10).to_numpy()
