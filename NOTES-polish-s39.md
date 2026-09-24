# S39 打磨六項（唔升指紋）

1. 和局：S26b 已證實不可用標籤規則改預測。產品只准「三擁接近」章。見 `data/research/s26/DRAW_DISPLAY_ONLY.md`。
2. 覆蓋解釋：`scripts/research/coverage_explain.py` 讀凍結 log；站點 `/football/explain` 讀 `hit_rate.json` 綠燈窗。
3. LGB SHAP：`scripts/research/lgb_shap_read_only.py` 只讀 `models/s5`，唔寫凍結。
4. Δλ：結構已有；寫入器強制 `delta=0 applied=false`。
5. 球隊頁 502：補 `/api/public/football-league`（積分由賽果 CSV，ELO 不造假）。
6. 禁止列不動：賠率入模、完場 xG 倒算、回測充場、NOTES-rejected.md 各刀。
