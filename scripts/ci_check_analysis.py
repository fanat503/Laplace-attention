# Copyright 2026 Slyatski Ilya
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


"""CI helper: assert the checkpoint-analysis JSON contains the paper probes."""
from __future__ import annotations

import argparse
import json
import math
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_json")
    args = ap.parse_args()
    d = json.load(open(args.analysis_json, encoding="utf-8"))
    required = [
        "val_loss", "induction_by_distance", "knockout_by_context_length",
        "prefix_matching", "attention_entropy",
    ]
    missing = [k for k in required if k not in d]
    if missing:
        sys.exit(f"analysis JSON missing keys: {missing}")
    if not math.isfinite(float(d["val_loss"])):
        sys.exit(f"non-finite val_loss in analysis: {d['val_loss']}")
    ko = d["knockout_by_context_length"]
    if not ko or not all("loss_full" in v for v in ko.values()):
        sys.exit("knockout_by_context_length malformed")
    print(f"ANALYSIS JSON OK: val_loss={float(d['val_loss']):.4f}, "
          f"knockout lengths={sorted(ko)}")


if __name__ == "__main__":
    main()
