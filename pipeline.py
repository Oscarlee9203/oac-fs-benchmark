"""
OAC feature-selection benchmark: data loaders, differential expression, DAV,
classical selectors and cross-validated AUC.
"""
import numpy as np, pandas as pd, gzip, re
from scipy import stats
import scipy.special as sp

RNG = np.random.default_rng(0)

# ---------- sex-linked genes: 12 Y-linked genes and XIST ----------
Y_GENES = ["RPS4Y1","DDX3Y","UTY","KDM5D","USP9Y","EIF1AY","NLGN4Y","TXLNGY","ZFY",
           "PRKY","TMSB4Y","RPS4Y2","XIST"]
SEX_Y_MARKERS = ["RPS4Y1","DDX3Y","UTY","KDM5D"]   # Y-linked markers used to infer sex (those present)
def load_sexchr_set():
    return set(Y_GENES)

# ============================================================
# Loaders
# ============================================================
def _read_counts_txt(path, drop_cols):
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={df.columns[0]:"symbol"})
    df = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore")
    df = df.groupby("symbol").mean(numeric_only=True)  # collapse dup symbols
    return df

def load_gse114007(datadir):
    oa = _read_counts_txt(f"{datadir}/GSE114007_OA_normalized.counts.txt.gz",
                          drop_cols=["Average OA","Max"])
    nm = _read_counts_txt(f"{datadir}/GSE114007_normal_normalized.counts.txt.gz",
                          drop_cols=["Average Normal","Max"])
    genes = oa.index.intersection(nm.index)
    oa, nm = oa.loc[genes], nm.loc[genes]
    X = pd.concat([oa, nm], axis=1)            # genes x samples
    y = np.array([1]*oa.shape[1] + [0]*nm.shape[1])
    return X.T.values.astype(float), np.array(X.index), y   # samples x genes, gene names, labels

# ============================================================
# Differential expression: empirical-Bayes moderated t (limma eBayes, 2 groups)
# ============================================================
def moderated_ttest(X, y):
    """X: samples x genes; y in {0,1}. Returns t, pval, fdr (BH)."""
    g1, g0 = X[y==1], X[y==0]
    n1, n0 = g1.shape[0], g0.shape[0]
    m1, m0 = g1.mean(0), g0.mean(0)
    v1, v0 = g1.var(0, ddof=1), g0.var(0, ddof=1)
    df_res = n1 + n0 - 2
    s2 = ((n1-1)*v1 + (n0-1)*v0) / df_res        # pooled residual variance
    s2 = np.clip(s2, 1e-12, None)
    # eBayes: estimate prior d0, s0^2 from distribution of s2 (Smyth 2004, method of moments)
    z = np.log(s2)
    e = z - sp.digamma(df_res/2) + np.log(df_res/2)
    emean = e.mean()
    evar = e.var(ddof=1) - sp.polygamma(1, df_res/2)   # df_res scalar -> scalar
    if evar <= 0:
        d0 = np.inf; s0_2 = np.exp(emean)
    else:
        from scipy.optimize import brentq
        try:
            d0 = 2*brentq(lambda d: sp.polygamma(1, d/2) - evar, 1e-3, 1e6)
        except Exception:
            d0 = 1e6
        s0_2 = np.exp(emean + sp.digamma(d0/2) - np.log(d0/2))
    if np.isinf(d0):
        s_tilde2 = np.full_like(s2, s0_2)
        df_total = np.full_like(s2, 1e6)
    else:
        s_tilde2 = (d0*s0_2 + df_res*s2) / (d0 + df_res)
        df_total = np.full_like(s2, d0 + df_res)
    se = np.sqrt(s_tilde2 * (1/n1 + 1/n0))
    t = (m1 - m0) / se
    p = 2*stats.t.sf(np.abs(t), df_total)
    fdr = bh_fdr(p)
    return t, p, fdr

def bh_fdr(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p)
    ranked = p[order]*n/(np.arange(1,n+1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked,0,1)
    return out

# ============================================================
# Metrics
# ============================================================
def point_biserial_r2(X, y):
    """per-gene squared point-biserial correlation with binary y. X samples x genes."""
    yc = (y - y.mean())
    Xc = X - X.mean(0)
    num = (Xc * yc[:,None]).sum(0)
    den = np.sqrt((Xc**2).sum(0) * (yc**2).sum())
    r = np.divide(num, den, out=np.zeros_like(num), where=den>0)
    return r**2

def infer_sex(X, genes):
    gi = {g:i for i,g in enumerate(genes)}
    ys = [g for g in SEX_Y_MARKERS if g in gi]
    if "XIST" in gi and ys:
        xist = X[:, gi["XIST"]]
        ymean = X[:, [gi[g] for g in ys]].mean(1)
        return (xist < ymean).astype(int)   # 1=male (low XIST, high Y)
    return np.zeros(X.shape[0], dtype=int)

def var_percentile(X, idx):
    v = X.var(0)
    ranks = stats.rankdata(v)/len(v)*100
    return np.median(ranks[idx])

# ============================================================
# Selectors (unsupervised ones fit on full X unless noted)
# ============================================================
def sel_supervised_de(X, y, k=100):
    F,_ = f_oneway_vec(X, y); return np.argsort(F)[::-1][:k]
def sel_highvar(X, y=None, k=100):
    return np.argsort(X.var(0))[::-1][:k]
def sel_pca(X, y=None, k=100, npc=10):
    Xs = (X - X.mean(0))/ (X.std(0)+1e-9)
    U,S,Vt = np.linalg.svd(Xs, full_matrices=False)
    load = np.abs(Vt[:npc]).max(0)
    return np.argsort(load)[::-1][:k]
def sel_random(X, y=None, k=100, seed=0):
    return np.random.default_rng(seed).choice(X.shape[1], k, replace=False)

def f_oneway_vec(X, y):
    g1, g0 = X[y==1], X[y==0]
    n1,n0=g1.shape[0],g0.shape[0]
    m1,m0=g1.mean(0),g0.mean(0); m=X.mean(0)
    ssb = n1*(m1-m)**2 + n0*(m0-m)**2
    ssw = ((g1-m1)**2).sum(0) + ((g0-m0)**2).sum(0)
    df_b,df_w=1,n1+n0-2
    F = (ssb/df_b)/np.clip(ssw/df_w,1e-12,None)
    p = stats.f.sf(F, df_b, df_w)
    return F, p

# ============================================================
# Classification AUC — 5x5 repeated stratified CV, L2 logistic
# ============================================================
def cv_auc(X, y, feat_idx=None, select_fn=None, k=100, n_splits=5, n_repeats=5, seed=0):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs=[]
    for tr,te in rskf.split(X,y):
        if select_fn is not None:              # leak-free: select inside train fold
            idx = select_fn(X[tr], y[tr], k)
        else:
            idx = feat_idx
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr][:,idx], y[tr])
        s = clf.predict_proba(X[te][:,idx])[:,1]
        aucs.append(roc_auc_score(y[te], s))
    return float(np.mean(aucs)), float(np.std(aucs))

# ---------- GSE117999 (Agilent) ----------
def load_gse117999(datadir):
    df = pd.read_excel(f"{datadir}/GSE117999_normalized_matrix.xlsx", sheet_name=0)
    ann = ["FeatureNum","ProbeUID","ControlType","ProbeName","GeneName"]
    samp = [c for c in df.columns if c not in ann]
    df = df[df["ControlType"]==0]                       # drop control probes
    df = df.dropna(subset=["GeneName"])
    df = df[~df[samp].isna().all(axis=1)]
    expr = df.groupby("GeneName")[samp].apply(lambda g: g.loc[g.mean(1).idxmax()])  # highest-mean probe/gene
    genes = np.array(expr.index)
    y = np.array([0 if s.startswith("P4-0") else 1 for s in samp])   # P4-1xx = OA (arthroplasty), P4-0xx = non-OA (meniscal tear); direction checked against the markers reported in the GEO record
    return expr.T.values.astype(float), genes, y, samp

# ---------- GSE57218 (Illumina HT-12, series matrix + bgx annotation) ----------
def _bgx_probe2sym(tarpath="data/GSE57218_RAW.tar"):
    import tarfile, gzip, io
    tf=tarfile.open(tarpath)
    m=[x for x in tf.getnames() if x.endswith(".bgx.gz")][0]
    raw=gzip.decompress(tf.extractfile(m).read()).decode("latin1").splitlines()
    hdr_i=next(i for i,l in enumerate(raw) if "Probe_Id" in l and "Symbol" in l)
    cols=raw[hdr_i].split("\t"); pi=cols.index("Probe_Id"); si=cols.index("Symbol")
    d={}
    for l in raw[hdr_i+1:]:
        if l.startswith("[") or not l.strip(): break
        f=l.split("\t")
        if len(f)>max(pi,si) and f[pi]: d[f[pi]]=f[si]
    return d

def load_gse57218(datadir):
    import io
    path=f"{datadir}/GSE57218_series_matrix.txt.gz"
    lines=gzip.open(path,"rt",encoding="latin1").read().splitlines()
    # phenotype
    def prow(key):
        return next(l for l in lines if l.startswith(key)).split("\t")[1:]
    disease=[x.strip('"').split(":")[-1].strip() for x in prow("!Sample_characteristics_ch1") if "disease state" in x.lower()]
    # disease-state row specifically
    dstate=None
    for l in lines:
        if l.startswith("!Sample_characteristics_ch1") and "disease state" in l.lower():
            dstate=[x.strip('"').split(":")[-1].strip() for x in l.split("\t")[1:]]
    sexrow=None; agerow=None
    for l in lines:
        if l.startswith("!Sample_characteristics_ch1") and "Sex:" in l:
            sexrow=[x.strip('"').split(":")[-1].strip() for x in l.split("\t")[1:]]
        if l.startswith("!Sample_characteristics_ch1") and "age (yrs)" in l.lower():
            agerow=[float(x.strip('"').split(":")[-1]) for x in l.split("\t")[1:]]
    b=next(i for i,l in enumerate(lines) if "series_matrix_table_begin" in l)
    e=next(i for i,l in enumerate(lines) if "series_matrix_table_end" in l)
    tbl="\n".join(lines[b+1:e])
    df=pd.read_csv(io.StringIO(tbl), sep="\t")
    df=df.rename(columns={df.columns[0]:"probe"}).set_index("probe")
    p2s=_bgx_probe2sym(f"{datadir}/GSE57218_RAW.tar")
    df["sym"]=[p2s.get(p,"") for p in df.index]
    df=df[df["sym"]!=""]
    scols=[c for c in df.columns if c!="sym"]
    df["_m"]=df[scols].mean(1)
    best=df.groupby("sym")["_m"].idxmax()          # best probe per gene
    expr=df.loc[best.values, scols]; expr.index=best.index   # genes x samples
    genes=np.array(expr.index)
    y=np.array([1 if d=="OA" else 0 for d in dstate])   # OA vs (Preserved+Healthy)
    sex=np.array([0 if s=="Male" else 1 for s in sexrow]) if sexrow else None
    age=np.array(agerow) if agerow else None
    return expr.T.values.astype(float), genes, y, dict(sex=sex, age=age, dstate=dstate)

# ---------- General GEO series-matrix loader (Affy etc., probe-level) ----------
def load_series_matrix(path, field, case, ctrl, log2=True):
    import io, re
    def clean(s): return re.sub(r'[^\x20-\x7e]','',s).strip()
    lines=gzip.open(path,'rt',encoding='latin1').read().splitlines()
    target=None
    for l in lines:
        if not l.startswith("!Sample_characteristics_ch1"): continue
        vals=[clean(x.strip('"')) for x in l.split("\t")[1:]]
        if any(field.lower() in v.lower() for v in vals):
            target=[v.split(":",1)[-1].strip() if ":" in v else v for v in vals]; break
    if target is None: raise ValueError("field not found: "+field)
    def lab(v):
        vl=v.lower()
        if any(c.lower() in vl for c in case): return 1
        if any(c.lower() in vl for c in ctrl): return 0
        return -1
    y_all=np.array([lab(v) for v in target])
    b=next(i for i,l in enumerate(lines) if "series_matrix_table_begin" in l)
    e=next(i for i,l in enumerate(lines) if "series_matrix_table_end" in l)
    df=pd.read_csv(io.StringIO("\n".join(lines[b+1:e])),sep="\t")
    df=df.rename(columns={df.columns[0]:"probe"}).set_index("probe")
    keep=np.where(y_all>=0)[0]
    X=df.values.T[keep].astype(float); y=y_all[keep]
    if log2: X=np.log2(np.clip(X,1,None))
    return X, np.array(df.index), y, {"n_total":len(y_all)}

DATASETS = {
 "OA_cartilage_114007": ("gse114007",),
 "OA_cartilage_117999": ("gse117999",),
 "OA_cartilage_57218":  ("gse57218",),
 "OA_synovium_55235":   ("sm","GSE55235_series_matrix.txt.gz","disease state",["osteoarthr"],["healthy"]),
 "RA_synovium_55235":   ("sm","GSE55235_series_matrix.txt.gz","disease state",["rheumatoid"],["healthy"]),
 "OA_synovium_55457":   ("sm","GSE55457_series_matrix.txt.gz","clinical status",["osteoarthr"],["normal"]),
 "RA_synovium_55457":   ("sm","GSE55457_series_matrix.txt.gz","clinical status",["rheumatoid"],["normal"]),
 "AD_brain_5281":       ("sm","GSE5281_series_matrix.txt.gz","Disease State",["alzheim"],["normal"]),
}
def annotated_sex(key, datadir="data"):
    """Deposited sex (1 = male, 0 = female) in loader sample order, or None if not parsed."""
    if DATASETS[key][0] != "gse57218": return None
    sex = load_gse57218(datadir)[3]["sex"]          # loader codes 0 = male, 1 = female
    return None if sex is None else (1 - sex)

def load_any(key, datadir="data"):
    spec=DATASETS[key]
    if spec[0]=="gse114007": return load_gse114007(datadir)[:3]
    if spec[0]=="gse117999": return load_gse117999(datadir)[:3]
    if spec[0]=="gse57218":  r=load_gse57218(datadir); return r[0],r[1],r[2]
    _,fn,field,case,ctrl=spec
    return load_series_matrix(f"{datadir}/{fn}",field,case,ctrl)[:3]
