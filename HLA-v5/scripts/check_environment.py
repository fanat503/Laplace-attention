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


def parse_pins(path: str) -> Dict[str, str]:
    pins: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        m = re.match(r"^([A-Za-z0-9_\-]+)(?:\[[^\]]+\])?==([^;\s]+)", line)
        if m:
            pins[m.group(1).replace("-", "_")] = m.group(2)
    return pins


def package_version(import_name: str) -> str:
    mod = importlib.import_module(import_name)
    return str(getattr(mod, "__version__", "unknown"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", default="requirements_tpu.txt")
    ap.add_argument("--require-xla", action="store_true")
    args = ap.parse_args()

    print(f"python={sys.version}")
    if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
        raise RuntimeError("Recommended Python range for pinned torch_xla stack is >=3.10,<3.12")

    pins = parse_pins(args.requirements)
    observed: Dict[str, str] = {}
    for pkg, expected in pins.items():
        import_name = PACKAGE_IMPORTS.get(pkg, pkg)
        try:
            version = package_version(import_name)
        except Exception as e:
            if pkg == "torch_xla" and not args.require_xla:
                print(f"{pkg}: not importable ({e}); allowed because --require-xla not set")
                continue
            raise RuntimeError(f"Failed to import {pkg} via {import_name}: {e}") from e
        observed[pkg] = version
        print(f"{pkg}: installed={version} expected={expected}")
        if version != expected:
            raise RuntimeError(f"Version mismatch for {pkg}: installed={version}, expected={expected}")

    if "torch" in observed and "torch_xla" in observed and observed["torch"] != observed["torch_xla"]:
        raise RuntimeError(f"torch/torch_xla mismatch: {observed['torch']} vs {observed['torch_xla']}")
    print("ENVIRONMENT VALID")


if __name__ == "__main__":
    main()
