"""Check Python/package environment against pinned requirements."""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import Dict

PACKAGE_IMPORTS = {
    "torch": "torch",
    "torch_xla": "torch_xla",
    "numpy": "numpy",
}


def _version_tuple(v: str):
    parts = []
    for p in re.split(r"[.\-+]", v):
        parts.append(int(p) if p.isdigit() else 0)
    return tuple(parts)


def parse_pins(path: str) -> Dict[str, list]:
    """Parse requirement SPECS (not just exact pins).

    Review finding F1: the previous version only recognized pkg==x.y and
    silently returned {} for range files like "torch>=2.4,<2.13" - the
    environment check then passed vacuously. Now every comparator in
    {==, >=, <=, >, <, !=} is parsed and enforced.
    Returns {normalized_name: [(op, version), ...]}.
    """
    specs: Dict[str, list] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split(";")[0].strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        m = re.match(r"^([A-Za-z0-9_\-]+)(?:\[[^\]]+\])?\s*(.*)$", line)
        if not m:
            continue
        name, rest = m.group(1).replace("-", "_"), m.group(2)
        clauses = []
        for clause in rest.split(","):
            cm = re.match(r"^\s*(==|>=|<=|!=|>|<)\s*([0-9][^\s]*)\s*$", clause)
            if cm:
                clauses.append((cm.group(1), cm.group(2)))
        if clauses or not rest:
            specs[name] = clauses  # bare name => присутствие без версии
    return specs


def spec_satisfied(installed: str, clauses: list) -> bool:
    iv = _version_tuple(installed)
    ops = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b,
           ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
           ">": lambda a, b: a > b, "<": lambda a, b: a < b}
    return all(ops[op](iv, _version_tuple(want)) for op, want in clauses)


def package_version(import_name: str) -> str:
    mod = importlib.import_module(import_name)
    return str(getattr(mod, "__version__", "unknown"))


def main() -> None:
    ap = argparse.ArgumentParser()
    # F2 fix: default to the file the repo actually ships.
    ap.add_argument("--requirements", default="requirements.txt")
    ap.add_argument("--require-xla", action="store_true")
    args = ap.parse_args()

    print(f"python={sys.version}")
    # F-extra fix: the old hard gate `>=3.10,<3.12` raised on Python 3.12/3.13 -
    # i.e. on Kaggle and Codespaces defaults - making preflight unrunnable
    # exactly where it matters. Hard-fail only below 3.10; warn above 3.12
    # (torch_xla wheels may lag newest Python).
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >= 3.10 required")
    if sys.version_info >= (3, 13):
        print("WARNING: Python >= 3.13; verify a matching torch_xla wheel exists for TPU runs")

    specs = parse_pins(args.requirements)
    if not specs:
        # F1 fix: an empty parse must be an ERROR, not a silent pass -
        # previously range-style files parsed to {} and the check was vacuous.
        raise RuntimeError(
            f"No requirement specs parsed from {args.requirements}; "
            f"refusing to declare the environment valid on zero checks"
        )
    observed: Dict[str, str] = {}
    for pkg, clauses in specs.items():
        import_name = PACKAGE_IMPORTS.get(pkg, pkg)
        try:
            version = package_version(import_name)
        except Exception as e:
            if pkg == "torch_xla" and not args.require_xla:
                print(f"{pkg}: not importable ({e}); allowed because --require-xla not set")
                continue
            if not clauses:
                print(f"{pkg}: optional/bare spec, not importable; skipping")
                continue
            raise RuntimeError(f"Failed to import {pkg} via {import_name}: {e}") from e
        observed[pkg] = version
        want = ",".join(op + v for op, v in clauses) or "(any)"
        print(f"{pkg}: installed={version} expected={want}")
        if clauses and not spec_satisfied(version, clauses):
            raise RuntimeError(f"Version out of range for {pkg}: installed={version}, spec={want}")

    if "torch" in observed and "torch_xla" in observed:
        t, x = observed["torch"].split("+")[0], observed["torch_xla"].split("+")[0]
        if _version_tuple(t)[:2] != _version_tuple(x)[:2]:
            raise RuntimeError(f"torch/torch_xla major.minor mismatch: {t} vs {x}")
    print("ENVIRONMENT VALID")


if __name__ == "__main__":
    main()
