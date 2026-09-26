import sys,csv,glob,datetime,json
sys.path.insert(0,'/tmp/s9/tianxi-football-database/scripts')
import xg_blend_s9 as X
M=X.load('/tmp/s9/xg_all.csv')
odds={}
for f in glob.glob('/tmp/s9/tianxi-football-database/data/results/*.csv'):
    div=f.split('_')[-1][:-4]
    if div not in('E0','SP1','I1','D1','F1'):continue
    for r in csv.DictReader(open(f,encoding='utf-8',errors='ignore')):
        try:
            d=r['Date'];dd=datetime.datetime.strptime(d,'%d/%m/%Y' if len(d.split('/')[-1])==4 else '%d/%m/%y').date()
        except Exception:continue
        def g(*ks):
            try:return [float(r[k]) for k in ks]
            except Exception:return None
        odds[(div,dd,r.get('HomeTeam'),r.get('AwayTeam'))]=(g('MaxH','MaxD','MaxA') or g('BbMxH','BbMxD','BbMxA'),g('B365H','B365D','B365A'))
def probs(w):
    out=[];model,last={},None;L={r['lg'] for r in M}
    for i,r in enumerate(M):
        if r['d']<X.START:continue
        if last is None or (r['d']-last).days>=14:
            last=r['d'];model={lg:X.fit([x for x in M[:i] if x['lg']==lg],r['d'],w) for lg in L}
        mo=model.get(r['lg'])
        if not mo:continue
        a,df,ha=mo
        if r['h'] not in a or r['a'] not in a:continue
        P=X.matrix(min(5,max(.15,a[r['h']]*df[r['a']]*ha)),min(5,max(.15,a[r['a']]*df[r['h']])))
        ph=sum(P[i2][j] for i2 in range(9) for j in range(9) if i2>j);pd=sum(P[k][k] for k in range(9))
        res=0 if r['gh']>r['ga'] else 1 if r['gh']==r['ga'] else 2
        out.append((r,[ph,pd,1-ph-pd],res))
    return out
def bt(rows,which,thr,only=None,pmax=None):
    n=st=pl=0
    for r,p,res in rows:
        o=odds.get((r['lg'],r['d'],r['h'],r['a']))
        if not o or not o[which]:continue
        O=o[which]
        if min(O)<=1.01:continue
        for k in range(3):
            if only is not None and k!=only:continue
            if pmax is not None and p[k]>pmax:continue
            if p[k]*O[k]-1>=thr:
                n+=1;st+=1;pl+=(O[k]-1) if res==k else -1
    return dict(bets=n,yield_pct=round(100*pl/st,2) if st else None)
R={}
for w in(1.0,0.3):
    rows=probs(w);j=sum(1 for r,_,_ in rows if (r['lg'],r['d'],r['h'],r['a']) in odds)
    s={'matches':len(rows),'with_odds':j}
    for t in(0,.02,.05,.10):s[f'flat_best_thr{int(t*100):02d}']=bt(rows,0,t)
    for t in(0,.05,.10):s[f'flat_b365_thr{int(t*100):02d}']=bt(rows,1,t)
    for t in(.05,.10):s[f'draw_only_best_thr{int(t*100):02d}']=bt(rows,0,t,1,.30)
    R[f'w_goals={w}']=s;print(w,json.dumps(s),flush=True)
json.dump(R,open('/tmp/s9/out/vb.json','w'),indent=1)
