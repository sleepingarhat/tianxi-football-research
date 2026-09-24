#!/usr/bin/env python3
"""Coverage explain from frozen prediction_log + results.

Reads production files read-only. Writes only under data/research/.
Coverage = argmax 1X2 hit and whether FT score is inside frozen top-8 cells.
Not causality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def side(p):
    h, d, a = (p or [0, 0, 0])[:3]
    if d >= h and d >= a:
        return 1
    return 0 if h >= a else 2


def ftr_idx(ftr: str | None) -> int | None:
    return {"H": 0, "D": 1, "A": 2, "home": 0, "draw": 1, "away": 2}.get(ftr or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Frozen monthly log JSON")
    ap.add_argument("--out", default="data/research/s39/coverage_explain.json")
    args = ap.parse_args()
    raw = json.loads(Path(args.log).read_text())
    matches = raw.get("matches") or raw
    if isinstance(matches, dict):
        rows = list(matches.values())
    else:
        rows = list(matches)
    n = hit = top8 = green = 0
    for m in rows:
        res = m.get("result") or {}
        if res.get("hg") is None and res.get("ft_h") is None:
            continue
        if m.get("status") not in (None, "final", "green", "locked"):
            continue
        green += 1
        p = m.get("p") or [0, 0, 0]
        actual = ftr_idx(res.get("ftr") or res.get("FTR"))
        if actual is not None:
            n += 1
            if side(p) == actual:
                hit += 1
        score = None
        if res.get("hg") is not None:
            score = f"{int(res['hg'])}-{int(res['ag'])}"
        elif res.get("ft_h") is not None:
            score = f"{int(res['ft_h'])}-{int(res['ft_a'])}"
        cells = ((m.get("cs") or {}).get("top8")) or []
        if score and any((c.get("score") == score) for c in cells):
            top8 += 1
    out = {
        "ok": True,
        "disclaimer": "覆蓋統計，唔係勝出因果。紅燈不入戰績。",
        "settled": green,
        "argmax_n": n,
        "argmax_hits": hit,
        "argmax_hit_rate": (hit / n) if n else None,
        "cs_top8_hits": top8,
        "cs_top8_rate": (top8 / green) if green else None,
        "wrote_freeze": False,
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
