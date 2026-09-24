#!/usr/bin/env python3
"""Read-only TreeSHAP on production S5 LGB track.

Does not write data/predictions, models, snapshots, or fingerprints.
Explains the LGB 1X2 head only — not Dixon-Coles cells, not ensemble alpha.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FEATURE_NOTE = (
    "Features already contain elo_* / lam_* / dc_p*; SHAP will split credit "
    "across collinear inputs. Report by league x home/away. Never say 勝出原因."
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lgb", required=True, help="Read-only path to models/s5/lgb.txt")
    p.add_argument("--meta", required=True, help="Read-only path to models/s5/meta.json")
    p.add_argument("--rows", help="Optional CSV of as-of features; if omitted only print plan")
    p.add_argument("--out", default="data/research/s39/lgb_shap_plan.json")
    args = p.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    cols = meta.get("feature_cols") or []
    out = {
        "ok": True,
        "fingerprint": meta.get("fingerprint"),
        "alpha": meta.get("alpha"),
        "n_features": len(cols),
        "feature_cols": cols,
        "method": "TreeSHAP on LGB track only",
        "applied_to_freeze": False,
        "note": FEATURE_NOTE,
        "rows_provided": bool(args.rows),
        "lgb_exists": Path(args.lgb).exists(),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"wrote": str(dest), "fingerprint": out["fingerprint"]}, ensure_ascii=False))
    if args.rows:
        print("# next: pip install shap lightgbm && compute TreeSHAP; do not commit booster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
