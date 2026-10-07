"""Deep selectors: unsupervised gated autoencoder (DFS-AE), PERSIST-style supervised
gate, CADA-AE (disease-supervised + confounder-adversarial + sparse gate) and the
cross-cohort XC-CADA-AE variants. All accept dev='cuda' or 'cpu'."""
import numpy as np, torch, torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None

def _prep(X, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return torch.tensor(Xs, dtype=torch.float32, device=dev)

def sel_ae_unsup(X, k=100, seed=0, epochs=200, lam=5e-4, dev=DEVICE):
    """Unsupervised gated AE (DFS-AE essence) -> |gate| scores (numpy)."""
    xt = _prep(X, seed, dev); D = X.shape[1]
    gate = nn.Parameter(torch.ones(D, device=dev))
    enc = nn.Sequential(nn.Linear(D,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256,128), nn.ReLU()).to(dev)
    dec = nn.Sequential(nn.Linear(128,256), nn.ReLU(), nn.Linear(256,D)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters())+list(dec.parameters())+[gate], lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(); z = enc(xt*gate); loss = ((dec(z)-xt)**2).mean() + lam*gate.abs().sum()
        loss.backward(); opt.step()
    return gate.detach().abs().cpu().numpy()

def sel_persist(X, y, k=100, seed=0, epochs=250, lam=5e-4, dev=DEVICE):
    """PERSIST-style supervised gated selection -> (topk idx, |gate| scores)."""
    xt = _prep(X, seed, dev); D = X.shape[1]
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    gate = nn.Parameter(torch.ones(D, device=dev))
    net = nn.Sequential(nn.Linear(D,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256,1)).to(dev)
    opt = torch.optim.AdamW(list(net.parameters())+[gate], lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = bce(net(xt*gate).squeeze(1), yt) + lam*gate.abs().sum()
        loss.backward(); opt.step()
    g = gate.detach().abs().cpu().numpy(); return np.argsort(g)[::-1][:k], g

def sel_cadaae(X, y, conf, k=100, seed=0, epochs=250, lam=5e-4, alpha=3.0, gamma=1.0,
               beta_rec=1.0, adv_lambda=1.0, dev=DEVICE):
    """CADA-AE -> (topk idx, |gate| scores)."""
    xt = _prep(X, seed, dev); D = X.shape[1]
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    ct = torch.tensor(conf, dtype=torch.float32, device=dev)
    gate = nn.Parameter(torch.ones(D, device=dev))
    enc = nn.Sequential(nn.Linear(D,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256,128), nn.ReLU()).to(dev)
    dec = nn.Sequential(nn.Linear(128,256), nn.ReLU(), nn.Linear(256,D)).to(dev)
    dhead = nn.Linear(128,1).to(dev); chead = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1)).to(dev)
    params = list(enc.parameters())+list(dec.parameters())+list(dhead.parameters())+list(chead.parameters())+[gate]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); z = enc(xt*gate)
        d = dhead(z).squeeze(1); c = chead(GRL.apply(z, adv_lambda)).squeeze(1)
        loss = beta_rec*((dec(z)-xt)**2).mean() + alpha*bce(d,yt) + gamma*bce(c,ct) + lam*gate.abs().sum()
        loss.backward(); opt.step()
    g = gate.detach().abs().cpu().numpy(); return np.argsort(g)[::-1][:k], g

def train_xccada(cohorts, k=100, seed=0, epochs=200, alpha=3.0, gamma_dom=1.0,
                 lam=5e-4, adv=1.0, dev=DEVICE):
    """Cross-cohort domain-adversarial shared-gate selector. cohorts: list of (X,y)."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xs = [ (X-X.mean(0))/(X.std(0)+1e-8) for X,_ in cohorts ]; D = Xs[0].shape[1]; ndom=len(cohorts)
    Xa = torch.tensor(np.vstack(Xs), dtype=torch.float32, device=dev)
    ya = torch.tensor(np.concatenate([y for _,y in cohorts]), dtype=torch.float32, device=dev)
    doma = torch.tensor(np.concatenate([[i]*len(y) for i,(_,y) in enumerate(cohorts)]), dtype=torch.long, device=dev)
    gate = nn.Parameter(torch.ones(D, device=dev))
    enc = nn.Sequential(nn.Linear(D,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256,128), nn.ReLU()).to(dev)
    dhead = nn.Linear(128,1).to(dev); dom = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,ndom)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters())+list(dhead.parameters())+list(dom.parameters())+[gate], lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(); ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); z = enc(Xa*gate)
        loss = alpha*bce(dhead(z).squeeze(1), ya) + gamma_dom*ce(dom(GRL.apply(z,adv)), doma) + lam*gate.abs().sum()
        loss.backward(); opt.step()
    g = gate.detach().abs().cpu().numpy(); return np.argsort(g)[::-1][:k], g

def train_xccada2(cohorts, k=100, seed=0, epochs=300, alpha=5.0, gamma_dom=1.0,
                  beta_rec=0.3, lam=5e-4, adv=1.0, dev=DEVICE):
    """XC-CADA-AE v2: per-cohort disease heads + reconstruction + domain-adversarial."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xs = [ (X-X.mean(0))/(X.std(0)+1e-8) for X,_ in cohorts ]; D=Xs[0].shape[1]; ndom=len(cohorts)
    Xts = [torch.tensor(x, dtype=torch.float32, device=dev) for x in Xs]
    yts = [torch.tensor(y, dtype=torch.float32, device=dev) for _,y in cohorts]
    gate = nn.Parameter(torch.ones(D, device=dev))
    enc = nn.Sequential(nn.Linear(D,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256,128), nn.ReLU()).to(dev)
    dec = nn.Sequential(nn.Linear(128,256), nn.ReLU(), nn.Linear(256,D)).to(dev)
    dheads = nn.ModuleList([nn.Linear(128,1) for _ in cohorts]).to(dev)
    dom = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,ndom)).to(dev)
    params = list(enc.parameters())+list(dec.parameters())+list(dheads.parameters())+list(dom.parameters())+[gate]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4); bce=nn.BCEWithLogitsLoss(); ce=nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss=0; zs=[]
        for i,(xt,yt) in enumerate(zip(Xts,yts)):
            z=enc(xt*gate); zs.append(z)
            loss=loss+alpha*bce(dheads[i](z).squeeze(1),yt)+beta_rec*((dec(z)-xt)**2).mean()
        zall=torch.cat(zs,0)
        doma=torch.cat([torch.full((len(y),),i,dtype=torch.long,device=dev) for i,(_,y) in enumerate(cohorts)])
        loss=loss+gamma_dom*ce(dom(GRL.apply(zall,adv)),doma)+lam*gate.abs().sum()
        loss.backward(); opt.step()
    return gate.detach().abs().cpu().numpy()
