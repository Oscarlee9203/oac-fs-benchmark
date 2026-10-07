#!/usr/bin/env python3
"""
OAC feature-selection benchmark - master runner.

`python run_all.py` reproduces every analysis and figure in the article with the
settings used there:

  * semi-synthetic benchmark (f2): a random 3,000-gene subset of the cartilage
    background (--synthG 3000), 10 seeds (--seeds 10), deep selectors trained for
    250 epochs (--synth-epochs 250); the sensitivity step (f2sens) repeats it at 80 and
    500 epochs;
  * real-data analyses (f3, f4, confounder): deep selectors trained for 500 epochs
    (--epochs 500; the first XC-CADA-AE version uses half of that); the split-half
    control uses 40 random splits; random baselines use 50 random panels.

Other usage:
  python run_all.py --fast                        # quick CPU smoke test (not the article's numbers)
  python run_all.py --steps f2,f2sens             # synthetic benchmark + training-length sensitivity only
  python run_all.py --steps f2 --synthG 0         # synthetic benchmark on the full gene set

Data: place the GEO files listed in README.md in ./data/. Outputs -> ./results/.
Autoencoder-based numbers depend on the PyTorch version; the versions pinned in
requirements.txt reproduce the article.
"""
import argparse, os, json, time, warnings, functools, numpy as np
print = functools.partial(print, flush=True)
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
    """Data-loading check: group sizes, share of differentially expressed genes (moderated t,
    BH FDR < 0.05) and DAV of the supervised panel must equal the stored reference values."""
    print("\n== Data-loading check (three cartilage cohorts) ==")
    ref={"OA_cartilage_114007":(20,18,16.0,0.679),"OA_cartilage_117999":(10,10,6.3,0.738),
         "OA_cartilage_57218":(33,40,1.9,0.282)}; out={}
    for k in CART:
        X,g,y=P.load_any(k); _,_,fdr=P.moderated_ttest(X,y)
        de=P.sel_supervised_de(X,y); r2=P.point_biserial_r2(X,y)
        out[k]=dict(case=int((y==1).sum()),ctrl=int((y==0).sum()),
                    bgDE=round(100*(fdr<0.05).mean(),1),SupDE_R2dis=round(float(np.median(r2[de])),3))
        T=ref[k]; got=(out[k]["case"],out[k]["ctrl"],out[k]["bgDE"],out[k]["SupDE_R2dis"])
        ok=got[:2]==T[:2] and abs(got[2]-T[2])<=0.1 and abs(got[3]-T[3])<=0.002
        print(f"  {k}: cases/controls {got[0]}/{got[1]}, DE genes {got[2]}%, DAV {got[3]}  [{'ok' if ok else 'MISMATCH, expected '+str(T)}]")
        if not ok: raise SystemExit(f"Data-loading check failed for {k}: got {got}, expected {T}")
    json.dump(out, open(f"{a.outdir}/repro.json","w"), indent=1)

def _f2_compute(a, synth_epochs):
    """Semi-synthetic benchmark for one training length. Returns (panel A, panel B) dicts."""
    Xbg,_,yb=P.load_gse114007("data"); Xbg=Xbg[yb==0]; seeds=list(range(a.seeds)); dev=a.device; G=a.synthG
    METH=["SupervisedDE","HighVar","PCA","Random","DFS-AE","CADA-AE"]; FCS=[0.5,1.0,1.5]
    resA={m:{fc:[] for fc in FCS} for m in METH}; perm={fc:[] for fc in FCS}
    for fc in FCS:
        for s in seeds:
            X,y,conf,tm,cm=make_v2(Xbg,G=G,log2fc=fc,conf_amp=2.0,conf_corr=0.0,seed=s)
            for m in METH: resA[m][fc].append(average_precision_score(tm,synth_score(m,X,y,conf,s,synth_epochs,dev)))
            yp=default_rng(1000+s).permutation(y); perm[fc].append(average_precision_score(tm,P.f_oneway_vec(X,yp)[0]))
        print(f"  [{synth_epochs} epochs] A fc={fc} done")
    A={"fcs":FCS,"methods":{m:{fc:ci(resA[m][fc]) for fc in FCS} for m in METH},
       "perm_null":{fc:ci(perm[fc]) for fc in FCS}}
    METHB=["SupervisedDE","DFS-AE","PERSIST","CADA-AE"]; CORRS=[0.0,0.4,0.8]
    AU={m:{c:[] for c in CORRS} for m in METHB}; CN={m:{c:[] for c in CORRS} for m in METHB}
    for cc in CORRS:
        for s in seeds:
            X,y,conf,tm,cm=make_v2(Xbg,G=G,log2fc=1.0,conf_amp=3.0,conf_corr=cc,seed=s)
            for m in METHB:
                z=synth_score(m,X,y,conf,s,synth_epochs,dev)
                AU[m][cc].append(average_precision_score(tm,z)); CN[m][cc].append(100*cm[np.argsort(z)[::-1][:100]].mean())
        print(f"  [{synth_epochs} epochs] B corr={cc} done")
    B={"corrs":CORRS,"auprc":{m:{c:ci(AU[m][c]) for c in CORRS} for m in METHB},
       "contam":{m:{c:ci(CN[m][c]) for c in CORRS} for m in METHB}}
    return A,B

def step_f2(a):
    print("\n== F2 ground-truth benchmark ==")
    A,B=_f2_compute(a,a.synth_epochs)
    json.dump(A, open(f"{a.outdir}/f2ci_A.json","w"),indent=1)
    json.dump(B, open(f"{a.outdir}/f2ci_B.json","w"),indent=1)

SENS_EPOCHS=[80,250,500]
def step_f2sens(a):
    print("\n== F2 sensitivity to training length ==")
    out={}
    for ep in SENS_EPOCHS:
        if ep==a.synth_epochs and os.path.exists(f"{a.outdir}/f2ci_A.json"):
            A=json.load(open(f"{a.outdir}/f2ci_A.json")); B=json.load(open(f"{a.outdir}/f2ci_B.json"))
        else:
            A,B=_f2_compute(a,ep)
        out[str(ep)]={"A":A,"B":B}
    json.dump(out, open(f"{a.outdir}/f2_sensitivity.json","w"),indent=1)

N_RANDOM=50
def step_f3(a):
    print("\n== F3 cross-cohort transfer ==")
    rnd_out={}
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
        # A random panel does not depend on the cohort it is "selected" in, so its in-sample and
        # cross-cohort DAV coincide. N_RANDOM panels give the floor against which DAV is read;
        # the AUC entry stays that of the first panel (seed 0).
        r2={k:P.point_biserial_r2(*D[k][:2]) for k in keys}
        pm=[float(np.mean([np.median(r2[k][P.sel_random(D[k][0],seed=s)]) for k in keys])) for s in range(N_RANDOM)]
        summ["Random"]["in"]=summ["Random"]["cross"]=float(np.mean(pm))
        rnd_out[tag]={"panel_means":pm,"mean":float(np.mean(pm)),"min":float(np.min(pm)),"max":float(np.max(pm)),
                      "n_panels_below":{m:int(sum(v<summ[m]["cross"] for v in pm)) for m in METH if m!="Random"}}
        json.dump(summ, open(f"{a.outdir}/f3_{tag}.json","w"),indent=1); print(f"  {tag} done")
    json.dump(rnd_out, open(f"{a.outdir}/f3_random.json","w"),indent=1)

def step_f4(a):
    print("\n== F4 leave-one-cohort-out (against matched random panels) ==")
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
        # the same N_RANDOM random panels are scored in every held-out cohort
        rnd.append([davC(held,P.sel_random(D[x][0],seed=s)) for s in range(N_RANDOM)])
        print(f"  held-out {held.split('_')[-1]} done")
    # Like-for-like baseline: each random panel's mean over the three held-out cohorts, compared
    # with each method's mean over the same three cohorts (empirical one-sided p, add-one).
    pm=np.mean(np.array(rnd),0); mean={m:float(np.mean(v)) for m,v in rows.items()}
    below={m:int((pm<v).sum()) for m,v in mean.items()}
    pemp={m:float((1+(pm>=v).sum())/(N_RANDOM+1)) for m,v in mean.items()}
    json.dump({"per_fold":rows,"mean":mean,"random_per_fold_mean":[float(np.mean(r)) for r in rnd],
               "random_panel_means":[float(v) for v in pm],"random_mean":float(pm.mean()),
               "random_range":[float(pm.min()),float(pm.max())],
               "n_random_below":below,"p_empirical":pemp,"n_random":N_RANDOM},
              open(f"{a.outdir}/f4_xc.json","w"),indent=1)
    for m in mean: print(f"  {m:14s} {mean[m]:.3f}  above {below[m]}/{N_RANDOM} random panels (p={pemp[m]:.3f})")
    print(f"  random panels: mean {pm.mean():.3f}, range {pm.min():.3f}-{pm.max():.3f}")

N_SPLITS=40
def step_splithalf(a):
    print("\n== Within-cohort split-half control ==")
    D=_common(CART); out={}; floor={}
    for k in CART:
        X,y,_=D[k]; vals=[]; rv=[]
        for s in range(N_SPLITS):
            rng=default_rng(s); A=[]
            for cls in (0,1):
                ix=np.where(y==cls)[0].copy(); rng.shuffle(ix); A+=list(ix[:len(ix)//2])
            A=set(A); B=np.array([i for i in range(len(y)) if i not in A]); A=np.array(sorted(A))
            r2B=P.point_biserial_r2(X[B],y[B])
            idx=P.sel_supervised_de(X[A],y[A]); vals.append(float(np.median(r2B[idx])))
            rv.append(float(np.median(r2B[P.sel_random(X,seed=s)])))   # random panel, same evaluation half
        out[k]=ci(vals); floor[k]=ci(rv)
        print(f"  {k.split('_')[-1]}: split-half DAV = {out[k][0]:.3f} (random panel in the same halves {floor[k][0]:.3f})")
    json.dump(out, open(f"{a.outdir}/splithalf.json","w"),indent=1)
    json.dump(floor, open(f"{a.outdir}/splithalf_random.json","w"),indent=1)

def step_confounder(a):
    print("\n== Real-data confounder (sex) capture - cartilage ==")
    D=_common(CART); rows={}; sset=P.load_sexchr_set()
    def summ(X,genes,r2d,r2s,idx):
        return {"R2dis":float(np.median(r2d[idx])),"R2sex":float(np.median(r2s[idx])),
                "sexchr":int(sum(genes[i] in sset for i in idx))}
    for k in CART:
        X,y,genes=D[k]; sex=P.infer_sex(X,genes); r2s=P.point_biserial_r2(X,sex); r2d=P.point_biserial_r2(X,y)
        gset=set(genes); ann=P.annotated_sex(k)
        rows[k]={"n_male":int(sex.sum()),"n":int(len(sex)),
                 "n_male_cases":int(sex[y==1].sum()),"n_cases":int((y==1).sum()),
                 "n_male_controls":int(sex[y==0].sum()),"n_controls":int((y==0).sum()),
                 "y_genes_used":[g for g in P.SEX_Y_MARKERS if g in gset],
                 "n_sexchr_genes_present":len(sset&gset),
                 "annotated_sex_agreement":None if ann is None else [int(((sex==1)==(ann==1)).sum()),int(len(ann))]}
        for m,idx in {"SupervisedDE":P.sel_supervised_de(X,y),"HighVar":P.sel_highvar(X)}.items():
            rows[k][m]=summ(X,genes,r2d,r2s,idx)
        # stochastic selectors: one entry per seed (deep selectors are single training runs per seed)
        rows[k]["Random"]=[summ(X,genes,r2d,r2s,P.sel_random(X,seed=s)) for s in range(a.conf_seeds)]
        rows[k]["DFS-AE"]=[summ(X,genes,r2d,r2s,np.argsort(M.sel_ae_unsup(X,seed=s,epochs=a.epochs,dev=a.device))[::-1][:100])
                           for s in range(a.conf_seeds)]
        rows[k]["CADA-AE"]=[summ(X,genes,r2d,r2s,M.sel_cadaae(X,y,sex,seed=s,epochs=a.epochs,dev=a.device)[0])
                            for s in range(a.conf_seeds)]
        print(f"  {k.split('_')[-1]} done")
    json.dump(rows, open(f"{a.outdir}/confounder_real.json","w"),indent=1)

def step_f5(a):
    print("\n== F5 multi-disease panel ==")
    rows=[]
    for name,key in PANEL:
        X,g,y=P.load_any(key); r2=P.point_biserial_r2(X,y); o={"name":name,"n":int(len(y))}
        # In-sample DAV uses the panel selected on all samples. For the AUC, data-dependent
        # selectors are re-fitted inside every training fold so that no test-fold information
        # enters the selection; a random panel does not depend on the data.
        spec=[("SupDE",P.sel_supervised_de(X,y),lambda Xt,yt,k: P.sel_supervised_de(Xt,yt,k)),
              ("HighVar",P.sel_highvar(X),lambda Xt,yt,k: P.sel_highvar(Xt,k=k)),
              ("Random",P.sel_random(X,seed=0),None)]
        for m,idx,fn in spec:
            o[m+"_DAV"]=float(np.median(r2[idx]))
            o[m+"_AUC"]=float(P.cv_auc(X,y,select_fn=fn)[0] if fn else P.cv_auc(X,y,feat_idx=idx)[0])
        rows.append(o); print(f"  {name} done")
    json.dump(rows, open(f"{a.outdir}/f5_panel.json","w"),indent=1)

def step_figs(a):
    print("\n== Figures ==")
    import figures; figures.make_all(a.outdir)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",default="repro,f2,f2sens,f3,f4,splithalf,confounder,f5,figs")
    ap.add_argument("--seeds",type=int,default=10)
    ap.add_argument("--epochs",type=int,default=500)
    ap.add_argument("--synth-epochs",dest="synth_epochs",type=int,default=250)
    ap.add_argument("--synthG",type=int,default=3000)
    ap.add_argument("--conf-seeds",dest="conf_seeds",type=int,default=5)
    ap.add_argument("--device",default=M.DEVICE)
    ap.add_argument("--outdir",default="results")
    ap.add_argument("--fast",action="store_true")
    a=ap.parse_args()
    if a.fast:
        a.seeds=3; a.epochs=100; a.synth_epochs=80; a.synthG=3000; a.conf_seeds=2
        global SENS_EPOCHS; SENS_EPOCHS=[40,80]
    os.makedirs(a.outdir,exist_ok=True)
    print(f"device={a.device} seeds={a.seeds} epochs={a.epochs} synthG={a.synthG or 'full'}")
    steps={"repro":step_repro,"f2":step_f2,"f2sens":step_f2sens,"f3":step_f3,"f4":step_f4,"splithalf":step_splithalf,
           "confounder":step_confounder,"f5":step_f5,"figs":step_figs}
    t0=time.time()
    for s in a.steps.split(","):
        s=s.strip()
        if s in steps: steps[s](a)
    print(f"\nAll done in {round(time.time()-t0)}s. Results in {a.outdir}/")

if __name__=="__main__":
    main()
