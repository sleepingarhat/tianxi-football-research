#!/usr/bin/env python3
"""Football knife-2: TreeSHAP on production S5 LGB only. No freeze writes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

NOTE = "Explains LGB 1X2 head only. elo_*/lam_*/dc_p* are collinear. Never say 勝出原因."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--rows")
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--out", default="data/research/s39/lgb_shap.json")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    cols = list(meta.get("feature_cols") or [])
    out = {
        "ok": True,
        "knife": 2,
        "sport": "football",
        "status": "blocked",
        "fingerprint": meta.get("fingerprint"),
        "alpha": meta.get("alpha"),
        "method": "TreeSHAP on LGB track",
        "applied_to_freeze": False,
        "note": NOTE,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_features": len(cols),
        "gain": [],
        "global_mean_abs": [],
        "reason": None,
    }
    lgb_path = Path(args.lgb)
    if not lgb_path.exists():
        out["reason"] = "models/s5/lgb.txt not readable here"
    else:
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError as e:
            out["reason"] = f"deps missing: {e}"
        else:
            booster = lgb.Booster(model_file=str(lgb_path))
            names = booster.feature_name() or cols
            gain = booster.feature_importance(importance_type="gain")
            pairs = sorted(zip(names, gain.tolist()), key=lambda x: -x[1])
            out["gain"] = [{"feature": n, "gain": float(g)} for n, g in pairs[:30]]
            rows_p = Path(args.rows) if args.rows else None
            if rows_p and rows_p.exists():
                try:
                    import pandas as pd
                    import shap
                except ImportError as e:
                    out["status"] = "partial"
                    out["reason"] = f"gain ok; shap deps missing: {e}"
                else:
                    df = pd.read_csv(rows_p)
                    use = [c for c in names if c in df.columns]
                    X = df[use].astype(float).fillna(0.0).to_numpy()[: args.max_rows]
                    sv = shap.TreeExplainer(booster).shap_values(X)
                    if isinstance(sv, list):
                        sv = np.stack(sv, axis=-1)
                    sv = np.asarray(sv)
                    # multiclass: (n_rows, n_features, n_class) -> 平均埋三類
                    mean_abs = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
                    order = np.argsort(-mean_abs)
                    out["global_mean_abs"] = [
                        {"feature": use[i], "mean_abs": float(mean_abs[i])}
                        for i in order[:20]
                    ]
                    out["status"] = "ok"
                    out["n_rows"] = int(len(X))
            else:
                out["status"] = "partial"
                out["reason"] = "gain from frozen booster; TreeSHAP needs as-of feature rows"

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"wrote": str(dest), "status": out["status"], "reason": out.get("reason")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
