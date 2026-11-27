"""Verify Lean 4.36.0 `LabAskWhyLean` axioms and >= 100-case differential parity with Python.

1. Runs `lake build` inside `lean/`.
2. Runs `#print axioms` via `lake env lean` on all 21 theorems in `LabAskWhyLean` and
   verifies that ONLY standard Lean axioms (`propext`, `Classical.choice`, `Quot.sound`)
   appear (zero `sorryAx` and zero custom/user axioms).
3. Generates 160 deterministic boundary and randomized test cases across all 4 formal
   subsystems (`checkAskWhy`, `checkEpistemicGraph`, `attributeAssayOutcome`, and
   `conformalStanceSetPermille`), evaluates them in Lean 4 via `#eval`, and asserts
   100% exact case-by-case agreement with Python (`src/lab_ask_why/schema.py`).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_DIR = REPO_ROOT / "lean"
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lab_ask_why.schema import (  # noqa: E402
    attribute_assay_outcome_permille,
    check_ask_why_permille,
    check_epistemic_graph_permille,
    conformal_stance_set_permille,
)

ALLOWED_LEAN_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

ALL_THEOREMS: tuple[str, ...] = (
    "LabAskWhyLean.firstViolation_nil",
    "LabAskWhyLean.firstViolation_cons",
    "LabAskWhyLean.firstViolation_none_iff",
    "LabAskWhyLean.checkAskWhy_ok_iff",
    "LabAskWhyLean.contested_claim_requires_experiment",
    "LabAskWhyLean.below_noinfo_always_abstains",
    "LabAskWhyLean.empty_rationale_always_rejected",
    "LabAskWhyLean.floor_permille_preserves_abstention",
    "LabAskWhyLean.corrupted_run_never_poisons_memory",
    "LabAskWhyLean.unidentifiable_spectrum_never_certified",
    "LabAskWhyLean.ceil_permille_preserves_violation",
    "LabAskWhyLean.certified_negative_sound",
    "LabAskWhyLean.certified_positive_sound",
    "LabAskWhyLean.firstMapViolation_none_iff",
    "LabAskWhyLean.checkEpistemicGraph_none_iff",
    "LabAskWhyLean.entailment_conflict_requires_experiment",
    "LabAskWhyLean.entailment_conflict_without_experiment_rejected",
    "LabAskWhyLean.mutual_exclusion_conflict_requires_experiment",
    "LabAskWhyLean.conformalStanceSet_nonempty",
    "LabAskWhyLean.conformal_quantile_covers_support",
    "LabAskWhyLean.dual_conformal_evidence_emits_conflict_set",
)

_DEPENDS_RE = re.compile(r"^'([^']+)' depends on axioms: \[([^\]]*)\]\s*$")
_NO_AXIOMS_RE = re.compile(r"^'([^']+)' does not depend on any axioms\s*$")


def _resolve_lake_bin(explicit_lake: str | None = None) -> str:
    if explicit_lake:
        return explicit_lake
    found = shutil.which("lake")
    if found:
        return found
    win_default = Path.home() / ".elan" / "bin" / ("lake.exe" if os.name == "nt" else "lake")
    if win_default.exists():
        return str(win_default)
    raise FileNotFoundError("Could not find `lake` binary on PATH or in ~/.elan/bin")


def _lean_bool(val: bool) -> str:
    return "true" if val else "false"


def _build_160_parity_cases(seed: int = 20261127) -> tuple[list[str], list[str]]:
    """Build 160 deterministic `(lean_expr, expected_python_str)` differential test cases."""
    rng = np.random.default_rng(seed)
    lean_exprs: list[str] = []
    py_expected: list[str] = []

    # --- Group 1: 40 `checkAskWhy` cases ---
    ask_cases: list[tuple[int, int, int, int, bool]] = [
        (349, 350, 1, 0, False),  # boundary floor abstain
        (350, 350, 1, 0, False),  # exact threshold pass
        (800, 350, 0, 0, True),   # empty rationale
        (800, 350, 2, 1, False),  # contested without experiment
        (800, 350, 2, 1, True),   # contested with experiment
        (0, 0, 0, 1, False),      # zero threshold contradict-only
    ]
    while len(ask_cases) < 40:
        top_pm = int(rng.integers(0, 1001))
        th_pm = int(rng.integers(100, 700))
        sup = int(rng.integers(0, 4))
        con = int(rng.integers(0, 4))
        has_exp = bool(rng.integers(0, 2))
        ask_cases.append((top_pm, th_pm, sup, con, has_exp))

    for top_pm, th_pm, sup, con, has_exp in ask_cases:
        py_res = check_ask_why_permille(
            top_pm / 1000.0,
            noinfo_thresh=th_pm / 1000.0,
            support_hits=sup,
            contradict_hits=con,
            has_discriminating_experiment=has_exp,
        )
        py_tag = "OK" if py_res is None else py_res["rule"]
        py_expected.append(f"ASK:{py_tag}")
        lean_exprs.append(
            f"(match LabAskWhyLean.checkAskWhy "
            f"{{ topScorePermille := {top_pm}, noinfoThreshPermille := {th_pm}, "
            f"supportHits := {sup}, contradictHits := {con}, "
            f"hasDiscriminatingExperiment := {_lean_bool(has_exp)} }} with "
            f"| none => \"ASK:OK\" | some v => \"ASK:\" ++ v.rule)"
        )

    # --- Group 2: 40 `checkEpistemicGraph` multi-claim cases ---
    for idx in range(40):
        n_nodes = int(rng.integers(2, 5))
        probes_py: list[dict[str, Any]] = []
        probes_lean: list[str] = []
        for _ in range(n_nodes):
            top_pm = int(rng.integers(300, 950))
            th_pm = 350
            sup = int(rng.integers(0, 3))
            con = int(rng.integers(0, 3))
            if idx % 3 != 0 and sup == 0 and con == 0:
                sup = 1
            has_exp = bool(rng.integers(0, 2))
            probes_py.append(
                {
                    "topScorePermille": top_pm,
                    "noinfoThreshPermille": th_pm,
                    "supportHits": sup,
                    "contradictHits": con,
                    "hasDiscriminatingExperiment": has_exp,
                }
            )
            probes_lean.append(
                f"{{ topScorePermille := {top_pm}, noinfoThreshPermille := {th_pm}, "
                f"supportHits := {sup}, contradictHits := {con}, "
                f"hasDiscriminatingExperiment := {_lean_bool(has_exp)} }}"
            )

        if idx == 0:
            imps = [(0, n_nodes + 1)]  # out-of-bounds implication test
            excs: list[tuple[int, int]] = []
        elif idx == 1:
            imps = []
            excs = [(0, n_nodes + 2)]  # out-of-bounds exclusion test
        else:
            imps = [(0, 1)] if rng.integers(0, 2) == 1 else []
            excs = [(0, n_nodes - 1)] if rng.integers(0, 2) == 1 else []

        py_g = check_epistemic_graph_permille(probes_py, implications=imps, exclusions=excs)
        py_gtag = "OK" if py_g is None else py_g["rule"]
        py_expected.append(f"GRAPH:{py_gtag}")

        imp_lean = ", ".join(f"{{ antecedent := {a}, consequent := {b} }}" for a, b in imps)
        exc_lean = ", ".join(f"{{ claimA := {a}, claimB := {b} }}" for a, b in excs)
        lean_exprs.append(
            f"(match LabAskWhyLean.checkEpistemicGraph "
            f"{{ probes := [{', '.join(probes_lean)}], "
            f"implications := [{imp_lean}], exclusions := [{exc_lean}] }} with "
            f"| none => \"GRAPH:OK\" | some v => \"GRAPH:\" ++ v.rule)"
        )

    # --- Group 3: 40 `attributeAssayOutcome` cases ---
    for idx in range(40):
        flags = [bool(rng.integers(0, 2) or idx % 2 == 0) for _ in range(6)]
        if idx < 20:
            flags = [True] * 6
            if idx < 6:
                flags[idx] = False
        res_pm = int(rng.integers(0, 220))
        max_res = 100
        ov_pm = int(rng.integers(0, 250))
        max_ov = 150
        if idx == 10:
            res_pm = 100
            ov_pm = 150
        sig = bool(rng.integers(0, 2))
        tel = {
            "preGateCertified": flags[0],
            "acousticNormal": flags[1],
            "scheduleWindowsRespected": flags[2],
            "tactileSeatedCleanly": flags[3],
            "posControlPassed": flags[4],
            "negControlBlankClean": flags[5],
            "spectralResidualPermille": res_pm,
            "maxResidualPermille": max_res,
            "precursorOverlapPermille": ov_pm,
            "maxOverlapPermille": max_ov,
            "signalAboveActivityThresh": sig,
        }
        py_out = attribute_assay_outcome_permille(tel)
        py_expected.append(f"ATTR:{py_out}")
        lean_exprs.append(
            f"(match LabAskWhyLean.attributeAssayOutcome "
            f"{{ preGateCertified := {_lean_bool(flags[0])}, "
            f"acousticNormal := {_lean_bool(flags[1])}, "
            f"scheduleWindowsRespected := {_lean_bool(flags[2])}, "
            f"tactileSeatedCleanly := {_lean_bool(flags[3])}, "
            f"posControlPassed := {_lean_bool(flags[4])}, "
            f"negControlBlankClean := {_lean_bool(flags[5])}, "
            f"spectralResidualPermille := {res_pm}, maxResidualPermille := {max_res}, "
            f"precursorOverlapPermille := {ov_pm}, maxOverlapPermille := {max_ov}, "
            f"signalAboveActivityThresh := {_lean_bool(sig)} }} with "
            f"| .executionArtifact => \"ATTR:executionArtifact\" "
            f"| .ambiguousCharacterization => \"ATTR:ambiguousCharacterization\" "
            f"| .certifiedPositive => \"ATTR:certifiedPositive\" "
            f"| .certifiedNegative => \"ATTR:certifiedNegative\")"
        )

    # --- Group 4: 40 `conformalStanceSetPermille` cases ---
    conf_Seeds: list[tuple[int, int, int, int]] = [
        (480, 470, 50, 700),   # dual conflict set {SUPPORT, CONTRADICT}
        (860, 80, 60, 680),    # singleton {SUPPORT}
        (70, 850, 80, 680),    # singleton {CONTRADICT}
        (40, 50, 910, 680),    # singleton {NEI}
        (334, 333, 333, 200),  # extreme low-quantile fallback tie
        (330, 340, 330, 200),  # extreme low-quantile fallback contradict
        (330, 330, 340, 200),  # extreme low-quantile fallback nei
    ]
    while len(conf_Seeds) < 40:
        raw_p = rng.dirichlet([1.0, 1.0, 1.0])
        ps = int(round(float(raw_p[0]) * 1000))
        pc = int(round(float(raw_p[1]) * 1000))
        pn = max(0, 1000 - ps - pc)
        q_pm = int(rng.integers(250, 920))
        conf_Seeds.append((ps, pc, pn, q_pm))

    for ps, pc, pn, q_pm in conf_Seeds:
        cs = conformal_stance_set_permille(ps, pc, pn, q_pm)
        py_expected.append(
            f"CONF:{int(cs['supportIn'])}:{int(cs['contradictIn'])}:{int(cs['neiIn'])}"
        )
        lean_exprs.append(
            f"(let cs := LabAskWhyLean.conformalStanceSetPermille {ps} {pc} {pn} {q_pm}; "
            f"s!\"CONF:{{if cs.supportIn then 1 else 0}}:"
            f"{{if cs.contradictIn then 1 else 0}}:"
            f"{{if cs.neiIn then 1 else 0}}\")"
        )

    return lean_exprs, py_expected


def run_lean_verification(
    *,
    lake_bin: str | None = None,
    clean_after: bool = False,
) -> dict[str, Any]:
    """Build `LabAskWhyLean`, audit `#print axioms`, and verify 160 `#eval` parity cases."""
    lake = _resolve_lake_bin(lake_bin)

    build_proc = subprocess.run(
        [lake, "build"],
        cwd=str(LEAN_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if build_proc.returncode != 0:
        raise RuntimeError(
            f"`lake build` failed (exit {build_proc.returncode}):\n"
            f"{build_proc.stdout}\n{build_proc.stderr}"
        )

    lean_exprs, py_expected = _build_160_parity_cases()
    assert len(lean_exprs) >= 100

    harness_lines = ["import LabAskWhyLean", ""]
    for thm in ALL_THEOREMS:
        harness_lines.append(f"#print axioms {thm}")
    harness_lines.append("")
    harness_lines.append("def parityResults : List String := [")
    harness_lines.append(",\n  ".join(lean_exprs))
    harness_lines.append("]")
    harness_lines.append('#eval IO.println ("PARITY_JSON:" ++ toString parityResults)')

    fd, tmp_lean_str = tempfile.mkstemp(suffix=".lean", prefix="_verify_parity_", dir=str(LEAN_DIR))
    os.close(fd)
    tmp_lean = Path(tmp_lean_str)
    try:
        tmp_lean.write_text("\n".join(harness_lines) + "\n", encoding="utf-8")
        eval_proc = subprocess.run(
            [lake, "env", "lean", tmp_lean.name],
            cwd=str(LEAN_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        tmp_lean.unlink(missing_ok=True)
        if clean_after:
            shutil.rmtree(LEAN_DIR / ".lake", ignore_errors=True)
            (LEAN_DIR / "lake-manifest.json").unlink(missing_ok=True)

    if eval_proc.returncode != 0:
        raise RuntimeError(
            f"`lake env lean` harness failed (exit {eval_proc.returncode}):\n"
            f"{eval_proc.stdout}\n{eval_proc.stderr}"
        )

    combined_lines = [
        line.strip()
        for line in (eval_proc.stdout + "\n" + eval_proc.stderr).splitlines()
        if line.strip()
    ]

    axiom_map: dict[str, list[str]] = {}
    disallowed_found: dict[str, list[str]] = {}
    parity_line: str | None = None

    for line in combined_lines:
        if line.startswith("PARITY_JSON:"):
            parity_line = line[len("PARITY_JSON:") :].strip()
            continue
        m_dep = _DEPENDS_RE.match(line)
        if m_dep:
            thm_name = m_dep.group(1)
            axioms = [a.strip() for a in m_dep.group(2).split(",") if a.strip()]
            axiom_map[thm_name] = axioms
            bad = [a for a in axioms if a not in ALLOWED_LEAN_AXIOMS]
            if bad:
                disallowed_found[thm_name] = bad
            continue
        m_none = _NO_AXIOMS_RE.match(line)
        if m_none:
            axiom_map[m_none.group(1)] = []

    missing_theorems = [t for t in ALL_THEOREMS if t not in axiom_map]
    if missing_theorems:
        raise RuntimeError(f"Missing `#print axioms` output for theorems: {missing_theorems}")
    if disallowed_found:
        raise RuntimeError(f"Disallowed / custom Lean axioms detected: {disallowed_found}")
    if parity_line is None:
        raise RuntimeError("Missing `PARITY_JSON:` line in `#eval` output")

    # Lean `toString (List String)` renders `[ASK:..., GRAPH:...]` without quotes around elements
    inner = parity_line.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    lean_results = [tok.strip() for tok in inner.split(",") if tok.strip()]

    if len(lean_results) != len(py_expected):
        raise RuntimeError(
            f"Parity case count mismatch: Lean returned {len(lean_results)} vs Python {len(py_expected)}"
        )

    mismatches: list[dict[str, Any]] = []
    for idx, (l_val, p_val) in enumerate(zip(lean_results, py_expected)):
        if l_val != p_val:
            mismatches.append({"case_index": idx, "lean": l_val, "python": p_val})

    if mismatches:
        raise RuntimeError(f"Lean-Python differential parity mismatches: {mismatches[:5]}")

    observed_axioms = sorted({ax for axs in axiom_map.values() for ax in axs})
    return {
        "theorems_checked": len(ALL_THEOREMS),
        "observed_axioms": observed_axioms,
        "allowed_axioms": sorted(ALLOWED_LEAN_AXIOMS),
        "zero_sorry_or_custom_axioms": True,
        "parity_cases_evaluated": len(py_expected),
        "parity_mismatches": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Verify Lean 4.36.0 axioms and >=100-case differential parity against Python."
    )
    p.add_argument("--lake", type=str, default=None, help="Explicit path to `lake` executable.")
    p.add_argument("--clean", action="store_true", help="Remove `lean/.lake` and manifest after check.")
    p.add_argument("--json", action="store_true", help="Emit JSON summary.")
    args = p.parse_args(argv)

    summary = run_lean_verification(lake_bin=args.lake, clean_after=bool(args.clean))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"[OK] Lean 4.36.0 verified: {summary['theorems_checked']} theorems, "
            f"axioms={summary['observed_axioms']} (0 sorryAx / 0 custom), "
            f"{summary['parity_cases_evaluated']}/{summary['parity_cases_evaluated']} Python parity cases matched."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
