#!/usr/bin/env python3
"""
OAC feature-selection benchmark — master runner (v2, GPU-ready).
Reproduces the 3-cohort validation and produces the analyses behind Figures 1-4
of the GigaScience manuscript, including the review-driven additions:
random baseline in leave-one-out, within-cohort split-half control, and the
real-data confounder (sex) analysis.

Usage:
  python run_all.py --fast                                   # quick CPU smoke test
  python run_all.py --steps repro,f2,f3,f4,splithalf,confounder,f5,figs \
                    --seeds 50 --epochs 500 --synthG 0 --device cuda

Data: place the 6 GEO datasets in ./data/ (see data/README.md). Outputs -> ./results/.
"""
import argparse, os, json, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import pipeline as P, methods as M
from numpy.random import default_rng
from sklearn.metrics import average_precision_score
from scipy import stats

def ci(a):
    a = np.asarray(a, float); m = a.mean()
    h = 1.96*a.std(ddof=1)/max(len(a),2)**0.5 if len(a) > 1 else 0.0
    return [float(m), float(m-h), float(m+h)]

# ---------------- semi-synthetic generator ----------------
def make_v2(Xbg, G=0, n_case=20, n_ctrl=20, n_true=50, n_conf=50,
            log2fc=1.0, conf_amp=2.0, conf_corr=0.0, seed=0):
    rng = default_rng(seed)
    if G and G < Xbg.shape[1]:
        gsub = rng.choice(Xbg.shape[1], G, replace=False); B = Xbg[:, gsub]
    else:
        B = Xbg; G = Xbg.shape[1]
    idx = rng.integers(0, B.shape[0], n_case+n_ctrl)
    X = B[idx].astype(float).copy() + rng.normal(0, B.std(0)*0.1+1e-6, (n_case+n_ctrl, G))
    y = np.array([1]*n_case + [0]*n_ctrl)
    perm = rng.permutation(G); true = perm[:n_true]; cg = perm[n_true:n_true+n_conf]
    X[np.ix_(y==1, true)] += log2fc
    ys = (y-y.mean())/(y.std()+1e-9)
    f = conf_corr*ys + np.sqrt(max(1-conf_corr**2, 0))*rng.normal(size=len(y))
    X[:, cg] += np.outer(f, rng.normal(size=n_conf)*conf_amp)
    tmask = np.zeros(G, bool); tmask[true] = True
    cmask = np.zeros(G, bool); cmask[cg] = True
    return X, y, (f > np.median(f)).astype(int), tmask, cmask

def synth_score(m, X, y, conf, s, epochs, dev):
    if m == "SupervisedDE": return P.f_oneway_vec(X, y)[0]
    if m == "HighVar":      return X.var(0)
    if m == "PCA":
        Xs=(X-X.mean(0))/(X.std(0)+1e-9); _,_,Vt=np.linalg.svd(Xs, full_matrices=False)
        return np.abs(Vt[:10]).max(0)
    if m == "Random":  return default_rng(s).random(X.shape[1])
    if m == "DFS-AE":  return M.sel_ae_unsup(X, seed=s, epochs=epochs, dev=dev)
    if m == "PERSIST": return M.sel_persist(X, y, seed=s, epochs=epochs, dev=dev)[1]
    if m == "CADA-AE": return M.sel_cadaae(X, y, conf, seed=s, epochs=epochs, dev=dev)[1]

CART = ["OA_cartilage_114007", "OA_cartilage_117999", "OA_cartilage_57218"]
PANEL = [("Cartilage-OA #1","OA_cartilage_114007"),("Cartilage-OA #2","OA_cartilage_117999"),
         ("Cartilage-OA #3","OA_cartilage_57218"),("Synovium-OA #1","OA_synovium_55235"),
         ("Synovium-OA #2","OA_synovium_55457"),("Synovium-RA #1","RA_synovium_55235"),
         ("Synovium-RA #2","RA_synovium_55457"),("Brain-AD","AD_brain_5281")]

def _common(keys):
    d={}; S=[]
    for k in keys: X,g,y=P.load_any(k); d[k]=(X,np.array(g),y); S.append(set(g))
    c=sorted(set.intersection(*S)); D={}
    for k in keys:
        X,g,y=d[k]; pos={gg:i for i,gg in enumerate(g)}; col=[pos[x] for x in c]
        D[k]=(X[:,col], y, np.array(c))
    return D

def _sel(D,k,method,epochs,dev):
    Xc,y,genes=D[k]
    if method=="SupervisedDE": return P.sel_supervised_de(Xc,y)
    if method=="HighVar":      return P.sel_highvar(Xc)
    if method=="PCA":          return P.sel_pca(Xc)
    if method=="Random":       return P.sel_random(Xc,seed=0)
    if method=="DFS-AE":       return np.argsort(M.sel_ae_unsup(Xc,seed=0,epochs=epochs,dev=dev))[::-1][:100]
    if method=="CADA-AE":      return M.sel_cadaae(Xc,y,P.infer_sex(Xc,genes),seed=0,epochs=epochs,dev=dev)[0]

# ---------------- steps ----------------
def step_repro(a):
    print("\n== Reproduction (3 OA cohorts vs internal reference) ==")
    ref={"OA_cartilage_114007":(20,18,15.8,0.679),"OA_cartilage_117999":(10,10,6.7,0.726),
         "OA_cartilage_57218":(33,40,2.6,0.278)}; out={}
    for k in CART:
        X,g,y=P.load_any(k); _,_,fdr=P.moderated_ttest(X,y)
        de=P.sel_supervised_de(X,y); r2=P.point_biserial_r2(X,y)
        out[k]=dict(case=int((y==1).sum()),ctrl=int((y==0).sum()),
                    bgDE=round(100*(fdr<0.05).mean(),1),SupDE_R2dis=round(float(np.median(r2[de])),3))
        T=ref[k]; print(f"  {k}: {out[k]['case']}/{out[k]['ctrl']} (ref {T[0]}/{T[1]}) "
                        f"bgDE {out[k]['bgDE']}% (ref {T[2]}) R2dis {out[k]['SupDE_R2dis']} (ref {T[3]})")
    json.dump(out, open(f"{a.outdir}/repro.json","w"), indent=1)

def step_f2(a):
    print("\n== F2 ground-truth benchmark ==")
    Xbg,_,yb=P.load_gse114007("data"); Xbg=Xbg[yb==0]; seeds=list(range(a.seeds)); dev=a.device; G=a.synthG
    METH=["SupervisedDE","HighVar","PCA","Random","DFS-AE","CADA-AE"]; FCS=[0.5,1.0,1.5]
    resA={m:{fc:[] for fc in FCS} for m in METH}; perm={fc:[] for fc in FCS}
    for fc in FCS:
        for s in seeds:
            X,y,conf,tm,cm=make_v2(Xbg,G=G,log2fc=fc,conf_amp=2.0,conf_corr=0.0,seed=s)
            for m in METH: resA[m][fc].append(average_precision_score(tm,synth_score(m,X,y,conf,s,a.synth_epochs,dev)))
            yp=default_rng(1000+s).permutation(y); perm[fc].append(average_precision_score(tm,P.f_oneway_vec(X,yp)[0]))
        print(f"  A fc={fc} done")
    json.dump({"fcs":FCS,"methods":{m:{fc:ci(resA[m][fc]) for fc in FCS} for m in METH},
               "perm_null":{fc:ci(perm[fc]) for fc in FCS}}, open(f"{a.outdir}/f2ci_A.json","w"),indent=1)
    METHB=["SupervisedDE","DFS-AE","PERSIST","CADA-AE"]; CORRS=[0.0,0.4,0.8]
    AU={m:{c:[] for c in CORRS} for m in METHB}; CN={m:{c:[] for c in CORRS} for m in METHB}
    for cc in CORRS:
        for s in seeds:
            X,y,conf,tm,cm=make_v2(Xbg,G=G,log2fc=1.0,conf_amp=3.0,conf_corr=cc,seed=s)
            for m in METHB:
                z=synth_score(m,X,y,conf,s,a.synth_epochs,dev)
                AU[m][cc].append(average_precision_score(tm,z)); CN[m][cc].append(100*cm[np.argsort(z)[::-1][:100]].mean())
        print(f"  B corr={cc} done")
    json.dump({"corrs":CORRS,"auprc":{m:{c:ci(AU[m][c]) for c in CORRS} for m in METHB},
               "contam":{m:{c:ci(CN[m][c]) for c in CORRS} for m in METHB}}, open(f"{a.outdir}/f2ci_B.json","w"),indent=1)

def step_f3(a):
    print("\n== F3 cross-cohort transfer ==")
    for tag,keys in [("cartilage",CART),("synovium",["OA_synovium_55235","OA_synovium_55457"])]:
        D=_common(keys); METH=["SupervisedDE","HighVar","PCA","Random","DFS-AE","CADA-AE"]; summ={}
        for m in METH:
            ins=[];cr=[];au=[]
            for x in keys:
                idx=_sel(D,x,m,a.epochs,a.device); ins.append(float(np.median(P.point_biserial_r2(*D[x][:2])[idx])))
                for b in keys:
                    if b==x: continue
                    cr.append(float(np.median(P.point_biserial_r2(*D[b][:2])[idx]))); au.append(P.cv_auc(D[b][0],D[b][1],feat_idx=idx)[0])
            summ[m]={"in":float(np.mean(ins)),"cross":float(np.mean(cr)),"cross_auc":float(np.mean(au))}
        json.dump(summ, open(f"{a.outdir}/f3_{tag}.json","w"),indent=1); print(f"  {tag} done")

def step_f4(a):
    print("\n== F4 leave-one-cohort-out (with Random & HighVar baselines) ==")
    D=_common(CART)
    def consensusDE(tr):
        Fs=[np.argsort(np.argsort(P.f_oneway_vec(*D[t][:2])[0])) for t in tr]
        return np.argsort(np.minimum.reduce(Fs))[::-1][:100]
    def davC(c,idx): return float(np.median(P.point_biserial_r2(*D[c][:2])[idx]))
    rows={k:[] for k in ["DE_single","DFS-AE_single","HighVar","ConsensusDE","XC-CADA-AE","XC-CADA-AE-v2"]}; rnd=[]
    for held in CART:
        tr=[k for k in CART if k!=held]; x=tr[0]
        rows["DE_single"].append(davC(held,P.sel_supervised_de(*D[x][:2])))
        rows["HighVar"].append(davC(held,P.sel_highvar(D[x][0])))
        rows["DFS-AE_single"].append(davC(held,np.argsort(M.sel_ae_unsup(D[x][0],seed=0,epochs=a.epochs,dev=a.device))[::-1][:100]))
        rows["ConsensusDE"].append(davC(held,consensusDE(tr)))
        rows["XC-CADA-AE"].append(davC(held,M.train_xccada([(D[k][0],D[k][1]) for k in tr],epochs=a.epochs//2,dev=a.device)[0]))
        g=np.mean([M.train_xccada2([(D[k][0],D[k][1]) for k in tr],epochs=a.epochs,seed=s,dev=a.device) for s in range(3)],0)
        rows["XC-CADA-AE-v2"].append(davC(held,np.argsort(g)[::-1][:100]))
        for s in range(50): rnd.append(davC(held,P.sel_random(D[x][0],seed=s)))
        print(f"  held-out {held.split('_')[-1]} done")
    rm=float(np.mean(rnd)); z=(np.mean(rows["ConsensusDE"])-rm)/np.std(rnd,ddof=1); pval=float(1-stats.norm.cdf(z))
    json.dump({"per_fold":rows,"mean":{m:float(np.mean(v)) for m,v in rows.items()},
               "random_mean":rm,"random_ci":ci(rnd)[1:],"p_consensus_gt_random":pval},
              open(f"{a.outdir}/f4_xc.json","w"),indent=1)
    print(f"  ConsensusDE {np.mean(rows['ConsensusDE']):.3f} vs Random {rm:.3f}  p={pval:.3f}")

def step_splithalf(a):
    print("\n== Within-cohort split-half control (winner's-curse) ==")
    D=_common(CART); out={}
    for k in CART:
        X,y,_=D[k]; vals=[]
        for s in range(a.seeds if a.seeds>=20 else 40):
            rng=default_rng(s); A=[]
            for cls in (0,1):
                ix=np.where(y==cls)[0].copy(); rng.shuffle(ix); A+=list(ix[:len(ix)//2])
            A=set(A); B=np.array([i for i in range(len(y)) if i not in A]); A=np.array(sorted(A))
            idx=P.sel_supervised_de(X[A],y[A]); vals.append(float(np.median(P.point_biserial_r2(X[B],y[B])[idx])))
        out[k]=ci(vals); print(f"  {k.split('_')[-1]}: within-cohort split-half DAV = {out[k][0]:.3f}")
    json.dump(out, open(f"{a.outdir}/splithalf.json","w"),indent=1)

def step_confounder(a):
    print("\n== Real-data confounder (sex) capture — cartilage ==")
    D=_common(CART); rows={}
    for k in CART:
        X,y,genes=D[k]; sex=P.infer_sex(X,genes); r2s=P.point_biserial_r2(X,sex); r2d=P.point_biserial_r2(X,y)
        sels={"SupervisedDE":P.sel_supervised_de(X,y),"HighVar":P.sel_highvar(X),"Random":P.sel_random(X,seed=0),
              "DFS-AE":np.argsort(M.sel_ae_unsup(X,seed=0,epochs=a.epochs,dev=a.device))[::-1][:100],
              "CADA-AE":M.sel_cadaae(X,y,sex,seed=0,epochs=a.epochs,dev=a.device)[0]}
        sset=P.load_sexchr_set()
        rows[k]={m:{"R2dis":float(np.median(r2d[idx])),"R2sex":float(np.median(r2s[idx])),
                    "sexchr":int(sum(genes[i] in sset for i in idx))} for m,idx in sels.items()}
        print(f"  {k.split('_')[-1]} done")
    json.dump(rows, open(f"{a.outdir}/confounder_real.json","w"),indent=1)

def step_f5(a):
    print("\n== F5 multi-disease panel ==")
    rows=[]
    for name,key in PANEL:
        X,g,y=P.load_any(key); r2=P.point_biserial_r2(X,y); o={"name":name,"n":int(len(y))}
        for m,idx in [("SupDE",P.sel_supervised_de(X,y)),("HighVar",P.sel_highvar(X)),("Random",P.sel_random(X,seed=0))]:
            o[m+"_DAV"]=float(np.median(r2[idx])); o[m+"_AUC"]=float(P.cv_auc(X,y,feat_idx=idx)[0])
        rows.append(o); print(f"  {name} done")
    json.dump(rows, open(f"{a.outdir}/f5_panel.json","w"),indent=1)

def step_figs(a):
    print("\n== Figures ==")
    import figures; figures.make_all(a.outdir)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",default="repro,f2,f3,f4,splithalf,confounder,f5,figs")
    ap.add_argument("--seeds",type=int,default=50)
    ap.add_argument("--epochs",type=int,default=500)
    ap.add_argument("--synth-epochs",dest="synth_epochs",type=int,default=250)
    ap.add_argument("--synthG",type=int,default=0)
    ap.add_argument("--device",default=M.DEVICE)
    ap.add_argument("--outdir",default="results")
    ap.add_argument("--fast",action="store_true")
    a=ap.parse_args()
    if a.fast: a.seeds=3; a.epochs=100; a.synth_epochs=80; a.synthG=3000
    os.makedirs(a.outdir,exist_ok=True)
    print(f"device={a.device} seeds={a.seeds} epochs={a.epochs} synthG={a.synthG or 'full'}")
    steps={"repro":step_repro,"f2":step_f2,"f3":step_f3,"f4":step_f4,"splithalf":step_splithalf,
           "confounder":step_confounder,"f5":step_f5,"figs":step_figs}
    t0=time.time()
    for s in a.steps.split(","):
        s=s.strip()
        if s in steps: steps[s](a)
    print(f"\nAll done in {round(time.time()-t0)}s. Results in {a.outdir}/")

if __name__=="__main__":
    main()
