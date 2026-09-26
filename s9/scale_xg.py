"""S9 駁接校準：PitchAPI xG 按重疊場次比率縮放到 Understat 口徑（只研究用）。"""
import glob, json, sys, pandas as pd
DB=sys.argv[1]; OUT=sys.argv[2]
u=pd.concat([pd.read_csv(f) for f in glob.glob(f"{DB}/data/xg_understat/*.csv")])
p=pd.concat([pd.read_csv(f) for f in glob.glob(f"{DB}/data/xg_pitchapi/*.csv")])
m=p.merge(u[["match_key","home_xg","away_xg"]],on="match_key",suffixes=("_p","_u"))
ratio={}
for div,g in m.groupby("div"):
    ratio[div]=float((g.home_xg_u.sum()+g.away_xg_u.sum())/(g.home_xg_p.sum()+g.away_xg_p.sum()))
allr=float((m.home_xg_u.sum()+m.away_xg_u.sum())/(m.home_xg_p.sum()+m.away_xg_p.sum()))
import os; os.makedirs(OUT,exist_ok=True)
for f in glob.glob(f"{DB}/data/xg/*.csv"):
    d=pd.read_csv(f); mk=d.source.eq("pitchapi")
    for c in ("home_xg","away_xg","home_npxg","away_npxg"):
        d.loc[mk,c]=d.loc[mk,c]*d.loc[mk,"div"].map(ratio)
    d.to_csv(f"{OUT}/{os.path.basename(f)}",index=False)
json.dump({"overlap_matches":len(m),"ratio_by_div":ratio,"ratio_all":allr},open(f"{OUT}/../scale.json","w"),indent=1)
print(len(m),allr,ratio)
