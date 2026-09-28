# tianxi-football-research

天喜足球「研究軌」倉庫。S24 分家後，研究同生產完全分離。

## 邊界（硬規則）

- 本倉**冇**凍結預測、冇 prediction_log、冇版本指紋、冇每日凍結流程。
- 本倉任何腳本、任何結果，**唔可以**寫入生產倉 `tianxi-football-database` 的
  `data/predictions/`、`models/`、`snapshots/` 或任何指紋欄。
- 研究要升級生產，唯一路徑：三主閘（RPS、實際比分格 log-loss、ECE）＋三副閘
  （大細 2.5 校準、對角總質量、頭八格覆蓋）逐季 walk-forward 全過，
  再由人手在生產倉開新指紋版本。研究倉自己升唔到指紋。
- 產品站（tianxi-site）只讀生產倉凍結檔，永不讀本倉。

## 結構

- `scripts/research/` — 試驗腳本（一次一把，唔准疊加）
- `data/research/` — 試驗結果 JSON（只增不改，保留失敗記錄）
- `NOTES-queue-e1.md` — 2026-09-27 聯合隊列結案

## 試驗 5 重開條件

S24 分家完成（✅）＋ Δλ 條件層結構落地（見生產倉 `docs/delta-schema.md`）
＋紅燈規則寫入本倉後，才可重開跑。
