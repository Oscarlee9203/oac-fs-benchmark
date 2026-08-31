"""Regenerate Figures 1-4 from results/*.json produced by run_all.py (v2)."""
import json, os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C={"SupervisedDE":"#0072B2","SupDE":"#0072B2","CADA-AE":"#D55E00","DFS-AE":"#009E73",
   "DFS-AE_single":"#009E73","PERSIST":"#CC79A7","HighVar":"#E69F00","PCA":"#56B4E9",
   "Random":"#000000","ConsensusDE":"#CC79A7","DE_single":"#0072B2",
   "XC-CADA-AE":"#E69F00","XC-CADA-AE-v2":"#8888cc"}
def _band(ax,xs,tr,c,lab,ls="-",mk="o"):
    m=[t[0] for t in tr];lo=[t[1] for t in tr];hi=[t[2] for t in tr]
    ax.plot(xs,m,marker=mk,color=c,label=lab,ls=ls,lw=2); ax.fill_between(xs,lo,hi,color=c,alpha=.18)

def f1(o):  # ground truth
    A=json.load(open(f"{o}/f2ci_A.json")); B=json.load(open(f"{o}/f2ci_B.json"))
    fig,ax=plt.subplots(1,3,figsize=(13.5,4.2)); fcs=A["fcs"]; fk=[str(f) for f in fcs]
    for m in ["SupervisedDE","CADA-AE","DFS-AE","HighVar","PCA","Random"]:
        ls="--" if m in ("HighVar","PCA","Random") else "-"; _band(ax[0],fcs,[A["methods"][m][k] for k in fk],C[m],m,ls=ls)
    ax[0].plot(fcs,[A["perm_null"][k][0] for k in fk],"k:",lw=1,label="random-ranking null")
    ax[0].set_title("A  Recovery of true genes (mean ± 95% CI)",fontsize=10)
    ax[0].set_xlabel("disease effect size (log2FC)"); ax[0].set_ylabel("AUPRC vs known truth"); ax[0].legend(fontsize=7,ncol=2); ax[0].grid(alpha=.3); ax[0].set_ylim(-.02,1)
    cs=B["corrs"]; ck=[str(c) for c in cs]
    for m in ["SupervisedDE","PERSIST","DFS-AE","CADA-AE"]: _band(ax[1],cs,[B["auprc"][m][k] for k in ck],C[m],m)
    ax[1].set_title("B  Recovery under confounding",fontsize=10); ax[1].set_xlabel("disease–confounder corr."); ax[1].set_ylabel("AUPRC"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    for m in ["SupervisedDE","PERSIST","DFS-AE","CADA-AE"]: _band(ax[2],cs,[B["contam"][m][k] for k in ck],C[m],m,mk="s")
    ax[2].set_title("C  Confounder contamination (lower better)",fontsize=10); ax[2].set_xlabel("disease–confounder corr."); ax[2].set_ylabel("% confounder in top-100"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{o}/Figure1_ground_truth.png",dpi=200,bbox_inches="tight"); plt.close()

def f2(o):  # universality
    rows=json.load(open(f"{o}/f5_panel.json")); names=[r["name"] for r in rows]; xi=np.arange(len(rows))
    fig,ax=plt.subplots(2,1,figsize=(9,6.6),sharex=True)
    for m,c in [("SupDE","#0072B2"),("HighVar","#E69F00"),("Random","#000000")]:
        ax[0].plot(xi,[r[m+"_AUC"] for r in rows],"o-",label=m,color=c)
    ax[0].set_ylim(0.5,1.03); ax[0].axhline(1,ls=":",color="gray"); ax[0].set_ylabel("classification AUC")
    ax[0].set_title("A  Classification AUC saturates for ALL methods (uninformative)",fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    w=.27
    for j,(m,c) in enumerate([("SupDE","#0072B2"),("HighVar","#E69F00"),("Random","#000000")]):
        ax[1].bar(xi+(j-1)*w,[r[m+"_DAV"] for r in rows],w,label=m,color=c)
    ax[1].set_ylabel("disease-attributable variance (DAV)"); ax[1].set_title("B  DAV separates supervised from random selection",fontsize=10)
    ax[1].set_xticks(xi); ax[1].set_xticklabels(names,rotation=30,ha="right",fontsize=8); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{o}/Figure2_universality.png",dpi=200,bbox_inches="tight"); plt.close()

def f3(o):  # transfer + split-half
    cart=json.load(open(f"{o}/f3_cartilage.json")); syn=json.load(open(f"{o}/f3_synovium.json"))
    sh=json.load(open(f"{o}/splithalf.json")) if os.path.exists(f"{o}/splithalf.json") else None
    fig,ax=plt.subplots(1,3,figsize=(14,4.3))
    for j,(dat,ttl) in enumerate([(cart,"A  Cartilage-OA (cross-platform)"),(syn,"B  Synovium-OA (same platform)")]):
        Ms=[m for m in ["SupervisedDE","CADA-AE","HighVar","DFS-AE","PCA","Random"] if m in dat]; x=np.arange(len(Ms)); w=.38
        ax[j].bar(x-w/2,[dat[m]["in"] for m in Ms],w,label="DAV in-sample",color="#0072B2")
        ax[j].bar(x+w/2,[dat[m]["cross"] for m in Ms],w,label="DAV cross-cohort",color="#D55E00")
        a2=ax[j].twinx(); a2.plot(x,[dat[m]["cross_auc"] for m in Ms],"k--o",ms=4); a2.set_ylim(0,1.05); a2.set_ylabel("cross AUC",fontsize=8)
        ax[j].set_xticks(x); ax[j].set_xticklabels(Ms,rotation=30,ha="right",fontsize=8); ax[j].set_ylabel("DAV"); ax[j].set_title(ttl,fontsize=10); ax[j].set_ylim(0,0.8)
        if j==0: ax[j].legend(loc="upper right",fontsize=7)
    if sh:
        ks=list(sh.keys()); labs=[k.split("_")[-1] for k in ks]; x=np.arange(len(ks)); w=.38
        ax[2].bar(x-w/2,[sh[k][0] for k in ks],w,yerr=[[sh[k][0]-sh[k][1] for k in ks],[sh[k][2]-sh[k][0] for k in ks]],capsize=3,label="within-cohort (split-half)",color="#009E73")
        ax[2].bar(x+w/2,[cart["SupervisedDE"]["cross"]]*len(ks),w,label="cross-cohort",color="#D55E00")
        ax[2].set_xticks(x); ax[2].set_xticklabels(labs,fontsize=8); ax[2].set_ylabel("held-out DAV (SupervisedDE)")
        ax[2].set_title("C  Within-cohort transfers, cross-cohort does not",fontsize=10); ax[2].set_ylim(0,0.7); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{o}/Figure3_transfer.png",dpi=200,bbox_inches="tight"); plt.close()

def f4(o):  # leave-one-out with random
    d=json.load(open(f"{o}/f4_xc.json")); pf=d["per_fold"]
    order=[m for m in ["DFS-AE_single","DE_single","HighVar","XC-CADA-AE","XC-CADA-AE-v2","ConsensusDE"] if m in pf]
    means=[np.mean(pf[m]) for m in order]
    fig,ax=plt.subplots(figsize=(7.2,4.4)); xb=np.arange(len(order))
    ax.bar(xb,means,color=[C[m] for m in order])
    rm,(rl,rh)=d["random_mean"],d["random_ci"]; ax.axhspan(rl,rh,color="gray",alpha=.25); ax.axhline(rm,color="gray",ls="--",lw=1.5,label=f"Random 95% CI [{rl:.3f},{rh:.3f}]")
    ax.set_xticks(xb); ax.set_xticklabels(order,rotation=25,ha="right",fontsize=8.5); ax.set_ylabel("held-out cohort DAV (leave-one-out)")
    ax.set_title("No method exceeds random for transferable selection\n(ConsensusDE vs Random: p=%.2f)"%d["p_consensus_gt_random"],fontsize=10)
    for i,m in enumerate(order): ax.text(i,means[i]+0.002,f"{means[i]:.3f}",ha="center",fontsize=8)
    ax.legend(fontsize=8); ax.set_ylim(0,max(means)*1.35)
    plt.tight_layout(); plt.savefig(f"{o}/Figure4_consensus.png",dpi=200,bbox_inches="tight"); plt.close()

def make_all(o="results"):
    for nm,fn in [("Figure1",f1),("Figure2",f2),("Figure3",f3),("Figure4",f4)]:
        try: fn(o); print(f"  {nm} saved")
        except FileNotFoundError as e: print(f"  {nm} skipped ({e})")
