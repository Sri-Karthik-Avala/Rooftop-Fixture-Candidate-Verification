# made by - Karthik
import os, sys, time, math, random, warnings, gc, tempfile
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

START = time.time()
SEED = int(os.environ.get("ERIS_SEED", 1234))
SMOKE = os.environ.get("ERIS_SMOKE", "0") == "1"
DEADLINE = float(os.environ.get("ERIS_DEADLINE", 3000))
EPOCHS = int(os.environ.get("ERIS_EPOCHS", 11))
BS = int(os.environ.get("ERIS_BS", 256))
LR = float(os.environ.get("ERIS_LR", 3e-4))
WD = float(os.environ.get("ERIS_WD", 3e-4))
FOLDS = int(os.environ.get("ERIS_FOLDS", 5))
MAXFOLDS = int(os.environ.get("ERIS_MAXFOLDS", FOLDS))
WORKERS = int(os.environ.get("ERIS_WORKERS", 4))
TTA = os.environ.get("ERIS_TTA", "1") == "1"
BACKBONES = os.environ.get("ERIS_BACKBONES", "efficientnet_b0:1.1:128,efficientnet_b0:1.3:112,resnet34:1.15:128").split(",")
CROPDIR = os.environ.get("ERIS_CROPDIR", "")
LABELS = [0.0, 1.0, 3.0]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
cv2.setNumThreads(0 if WORKERS > 0 else max(2, os.cpu_count() or 4))

if SMOKE:
    EPOCHS, FOLDS = 1, 2
    BACKBONES = ["efficientnet_b0:1.1:112"]

def seed_all(s):
    random.seed(s); np.random.seed(s); torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
seed_all(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def macro_f1(y_true, y_pred):
    scores = []
    for label in LABELS:
        tp = float(((y_true == label) & (y_pred == label)).sum())
        fp = float(((y_true != label) & (y_pred == label)).sum())
        fn = float(((y_true == label) & (y_pred != label)).sum())
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom == 0.0 else (2.0 * tp) / denom)
    return float(np.mean(scores))


def find_data_root():
    here = Path(__file__).resolve().parent
    cands = [here, Path("."), Path(".."), Path("dataset/public"), Path("../dataset/public"),
             Path("/kaggle/input"), here / "dataset" / "public"]
    for c in cands:
        try:
            subs = [c] + ([p for p in c.iterdir() if p.is_dir()] if c.is_dir() else [])
        except Exception:
            subs = [c]
        for sub in subs:
            if (sub / "train.csv").exists() and (sub / "test.csv").exists():
                return sub
    for base in [here, Path("/kaggle/input"), Path("/")]:
        try:
            for p in base.rglob("train.csv"):
                if (p.parent / "test.csv").exists():
                    return p.parent
        except Exception:
            pass
    raise FileNotFoundError("data root not found")


def resolve_image_path(root, ip, image_id):
    for cand in [root / str(ip), root / Path(str(ip)).name, root / "train" / f"{image_id}.jpg",
                 root / "test" / f"{image_id}.jpg", root / f"{image_id}.jpg"]:
        if cand.exists():
            return cand
    hits = list(root.rglob(f"{image_id}.jpg"))
    return hits[0] if hits else None


def load_images(root, df):
    imgs = {}
    for iid, sub in df.groupby("image_id"):
        r = sub.iloc[0]
        p = resolve_image_path(root, r["image_path"], iid)
        im = cv2.imread(str(p), cv2.IMREAD_COLOR) if p is not None else None
        if im is None:
            im = np.full((int(r["image_height"]), int(r["image_width"]), 3), 127, np.uint8)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs[iid] = np.ascontiguousarray(im)
    return imgs


def extract_crops(df, images, ctx, size, box, tag):
    p = int(round(ctx * box))
    cx = ((df["x_min"].values + df["x_max"].values) / 2.0).astype(np.float32)
    cy = ((df["y_min"].values + df["y_max"].values) / 2.0).astype(np.float32)
    ids = df["image_id"].values
    path = os.path.join(CROPDIR or tempfile.gettempdir(), f"eris_crops_{tag}_{os.getpid()}.npy")
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(df), size, size, 3))
    for i in range(len(df)):
        im = images[ids[i]]
        patch = cv2.getRectSubPix(im, (p, p), (float(cx[i]), float(cy[i])))
        if p != size:
            interp = cv2.INTER_AREA if p > size else cv2.INTER_CUBIC
            patch = cv2.resize(patch, (size, size), interpolation=interp)
        arr[i] = patch
    arr.flush()
    del arr
    return path


def radial_channel(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (size / 2.0)
    return np.clip(1.0 - r, -1.0, 1.0).astype(np.float32)


def augment(img):
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k).copy()
    if random.random() < 0.5:
        img = img[:, ::-1].copy()
    f = img.astype(np.float32)
    if random.random() < 0.9:
        c = random.uniform(0.7, 1.3); b = random.uniform(-18, 18); m = f.mean()
        f = (f - m) * c + m + b
    if random.random() < 0.4:
        g = random.uniform(0.8, 1.25)
        f = 255.0 * np.clip(f / 255.0, 0, 1) ** g
    f = np.clip(f, 0, 255).astype(np.uint8)
    r = random.random()  # at most one dot-preserving heavy degradation
    if r < 0.30:
        s = random.uniform(0.5, 0.85); h, w = f.shape[:2]; ns = max(12, int(round(h * s)))
        f = cv2.resize(f, (ns, ns), interpolation=cv2.INTER_AREA)
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
    elif r < 0.45:
        f = cv2.GaussianBlur(f, (3, 3), 0)
    elif r < 0.58:
        q = random.randint(40, 85)
        ok, enc = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            f = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if random.random() < 0.25:
        f = np.clip(f.astype(np.float32) + np.random.randn(*f.shape).astype(np.float32) * random.uniform(3, 9), 0, 255).astype(np.uint8)
    return f


class CropDS(Dataset):
    def __init__(self, crops_path, idx, geom, labels, size, train):
        self.crops_path = crops_path; self.idx = idx
        self.geom = geom; self.labels = labels; self.train = train; self.size = size
        self.rad = radial_channel(size); self._mm = None

    def _crops(self):
        if self._mm is None:
            self._mm = np.load(self.crops_path, mmap_mode="r")
        return self._mm

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = np.array(self._crops()[j])
        if self.train:
            img = augment(img)
        f = img.astype(np.float32) / 255.0
        f = (f - IMAGENET_MEAN) / IMAGENET_STD
        f = np.concatenate([f, self.rad[..., None]], axis=2)
        x = torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1)))
        g = torch.from_numpy(self.geom[j])
        y = -1 if self.labels is None else int(self.labels[j])
        return x, g, y


class Net(nn.Module):
    def __init__(self, backbone, n_geom):
        super().__init__()
        import timm
        try:
            self.body = timm.create_model(backbone, pretrained=True, num_classes=0,
                                          global_pool="avg", in_chans=4)
        except Exception as e:
            print(f"[warn] pretrained load failed for {backbone}: {e}; from-scratch fallback", flush=True)
            self.body = timm.create_model(backbone, pretrained=False, num_classes=0,
                                          global_pool="avg", in_chans=4)
        fdim = self.body.num_features
        self.geom = nn.Sequential(
            nn.Linear(n_geom, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.SiLU())
        self.geom_head = nn.Linear(64, 3)  # geometry-only aux logits (keeps geom pathway strong)
        self.head = nn.Sequential(nn.Linear(fdim + 64, 192), nn.SiLU(), nn.Dropout(0.3), nn.Linear(192, 3))

    def forward(self, x, g):
        ge = self.geom(g)
        fused = self.head(torch.cat([self.body(x), ge], 1))
        return fused, self.geom_head(ge)


def d4_variants(x):
    outs = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])]
    for k in (1, 3):
        r = torch.rot90(x, k, dims=[2, 3])
        outs += [r, torch.flip(r, dims=[3])]
    return outs


@torch.no_grad()
def predict(model, ds, tta):
    model.eval()
    dl = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=WORKERS, pin_memory=True)
    fused, geom = [], []
    for x, g, _ in dl:
        x = x.to(DEV, non_blocking=True); g = g.to(DEV, non_blocking=True)
        variants = d4_variants(x) if tta else [x]  # radial ch is D4-invariant -> stays aligned
        acc = 0; gp = None
        for vi, v in enumerate(variants):
            with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
                fo, go = model(v, g)
                acc = acc + F.softmax(fo.float(), dim=1)
                if vi == 0:
                    gp = F.softmax(go.float(), dim=1)
        fused.append((acc / len(variants)).cpu().numpy())
        geom.append(gp.cpu().numpy())
    return np.concatenate(fused), np.concatenate(geom)


def tune_weights(P, y):
    g0 = np.geomspace(0.1, 6.0, 30)
    g2 = np.geomspace(0.1, 10.0, 34)
    best, bw = -1.0, (1.0, 1.0, 1.0)
    ycls = np.array(LABELS)[y]
    for w0 in g0:
        for w2 in g2:
            pred = np.array(LABELS)[np.argmax(P * np.array([w0, 1.0, w2]), axis=1)]
            s = macro_f1(ycls, pred)
            if s > best:
                best, bw = s, (float(w0), 1.0, float(w2))
    return bw, best


def train_fold(crops_path, geom, y, tr_idx, va_idx, test_path, test_geom, backbone, size, seed):
    seed_all(seed)
    model = Net(backbone, geom.shape[1]).to(DEV)
    ytr = y[tr_idx]
    cls_count = np.array([(ytr == c).sum() for c in range(3)], np.float32)
    cls_w = (cls_count.sum() / (3 * np.maximum(cls_count, 1))) ** 0.5
    ce_w = torch.tensor(cls_w / cls_w.mean(), dtype=torch.float32, device=DEV)
    tr_ds = CropDS(crops_path, tr_idx, geom, y, size, train=True)
    va_ds = CropDS(crops_path, va_idx, geom, y, size, train=False)
    tr_dl = DataLoader(tr_ds, batch_size=BS, shuffle=True, num_workers=WORKERS, pin_memory=True,
                       drop_last=True, persistent_workers=(WORKERS > 0))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, len(tr_dl)) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))
    lossf = nn.CrossEntropyLoss(weight=ce_w, label_smoothing=0.05)
    for ep in range(EPOCHS):
        model.train()
        for x, g, yb in tr_dl:
            x = x.to(DEV, non_blocking=True); g = g.to(DEV, non_blocking=True); yb = yb.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
                out, gout = model(x, g)
                loss = lossf(out, yb) + 0.4 * lossf(gout, yb)
            scaler.scale(loss).backward()
            sb = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= sb:
                sched.step()
        if time.time() - START > DEADLINE:
            print(f"  [deadline in-fold @ epoch {ep}] finalizing this fold", flush=True)
            break
    oof, oof_g = predict(model, va_ds, TTA)
    test_ds = CropDS(test_path, np.arange(len(test_geom)), test_geom, None, size, train=False)
    tst, _ = predict(model, test_ds, TTA)
    del model, tr_dl; torch.cuda.empty_cache(); gc.collect()
    return oof, oof_g, tst


KNN_KS = [3, 5, 8, 12]
DENS_RS = [16, 24, 32, 48, 64, 96]
CENT_RS = [24, 48, 96]


def cluster_feats(df, box):
    # geometric candidate-verification features from per-image candidate coordinates (no labels, no test-set stats)
    N = len(df)
    cxp = ((df["x_min"].values + df["x_max"].values) / 2.0).astype(np.float32)
    cyp = ((df["y_min"].values + df["y_max"].values) / 2.0).astype(np.float32)
    W = df["image_width"].values.astype(np.float32); H = df["image_height"].values.astype(np.float32)
    ncols = 3 + len(KNN_KS) * 2 + len(DENS_RS) * 2 + len(CENT_RS) + 6
    out = np.zeros((N, ncols), np.float32)
    for iid, gi in df.groupby("image_id").indices.items():
        idx = np.asarray(gi)
        C = np.stack([cxp[idx], cyp[idx]], 1)
        n = len(idx)
        d = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)).astype(np.float32)
        np.fill_diagonal(d, 1e9)
        order = np.argsort(d, axis=1)
        sd = np.sort(d, axis=1)
        nnk5 = sd[:, :min(5, max(1, n - 1))].mean(1)
        col = np.zeros((n, ncols), np.float32)
        for a in range(3):
            col[:, a] = sd[:, min(a, n - 2)] if n > 1 else 0.0
        c0 = 3
        for j, k in enumerate(KNN_KS):
            kk = min(k, max(1, n - 1))
            col[:, c0 + j] = sd[:, :kk].mean(1)
            cen = C[order[:, :kk]].mean(1)
            col[:, c0 + len(KNN_KS) + j] = np.sqrt(((C - cen) ** 2).sum(1))
        c0 += 2 * len(KNN_KS)
        for j, R in enumerate(DENS_RS):
            nb = d < R
            cnt = nb.sum(1).astype(np.float32)
            col[:, c0 + j] = cnt
            csum = nb.astype(np.float32) @ C
            cen = csum / np.maximum(cnt[:, None], 1)
            off = np.sqrt(((C - cen) ** 2).sum(1))
            col[:, c0 + len(DENS_RS) + j] = np.where(cnt > 0, off, 0.0)
        c0 += 2 * len(DENS_RS)
        for j, R in enumerate(CENT_RS):
            nb = d < R
            # fraction of within-R neighbors whose nnk5 >= mine (higher => more central/tight)
            gt = (nnk5[None, :] >= nnk5[:, None]) & nb
            denom = np.maximum(nb.sum(1), 1)
            col[:, c0 + j] = gt.sum(1) / denom
        c0 += len(CENT_RS)
        col[:, c0 + 0] = np.log(W[idx]); col[:, c0 + 1] = np.log(H[idx])
        col[:, c0 + 2] = box / W[idx]; col[:, c0 + 3] = box / H[idx]
        col[:, c0 + 4] = cxp[idx] / W[idx]; col[:, c0 + 5] = cyp[idx] / H[idx]
        out[idx] = col
    out = np.nan_to_num(out, nan=0.0, posinf=1e4, neginf=0.0)
    out[:, :3 + len(KNN_KS) * 2 + len(DENS_RS) * 2] = np.minimum(
        out[:, :3 + len(KNN_KS) * 2 + len(DENS_RS) * 2], 1e4)
    return out


def build_geom(df, box, stats=None):
    feats = cluster_feats(df, box)
    nd = 3 + len(KNN_KS) * 2 + len(DENS_RS) * 2  # log1p the distance/count magnitude columns
    feats[:, :nd] = np.log1p(np.clip(feats[:, :nd], 0, None))
    if stats is None:
        stats = (feats.mean(0), feats.std(0) + 1e-6)
    mu, sd = stats
    return ((feats - mu) / sd).astype(np.float32), stats


def write_sub(test_df, labels, path):
    pd.DataFrame({"image_id": test_df["image_id"].values, "candidate_id": test_df["candidate_id"].values,
                  "predicted_relevance": labels}).to_csv(path, index=False)


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    root = find_data_root()
    print("data root:", root, flush=True)
    train_df = pd.read_csv(root / "train.csv")
    test_df = pd.read_csv(root / "test.csv")
    if SMOKE:
        train_df = train_df[train_df["image_id"].isin(train_df["image_id"].drop_duplicates().head(40))].reset_index(drop=True)
        test_df = test_df[test_df["image_id"].isin(test_df["image_id"].drop_duplicates().head(20))].reset_index(drop=True)

    os.makedirs("working", exist_ok=True)
    subpath = "working/submission.csv"
    write_sub(test_df, np.full(len(test_df), 1.0), subpath)  # safety-net file before any column-dependent op
    print(f"train rows {len(train_df)} test rows {len(test_df)}", flush=True)

    box = int(round(float(np.median(train_df["x_max"].values - train_df["x_min"].values))))
    print("derived box size:", box, flush=True)
    y = train_df["relevance"].map({0.0: 0, 1.0: 1, 3.0: 2}).values.astype(np.int64)
    groups = train_df["image_id"].values
    geom_tr, gstats = build_geom(train_df, box)
    geom_te, _ = build_geom(test_df, box, gstats)

    tr_images = load_images(root, train_df)
    te_images = load_images(root, test_df)
    print(f"loaded {len(tr_images)} train imgs, {len(te_images)} test imgs, {time.time()-START:.0f}s", flush=True)

    n = len(train_df)
    oof_sum = np.zeros((n, 3), np.float32); oof_cnt = np.zeros(n, np.float32)
    oof_gsum = np.zeros((n, 3), np.float32)
    test_sum = np.zeros((len(test_df), 3), np.float32); test_cnt = 0.0
    configs = []
    for spec in BACKBONES:
        pr = spec.split(":")
        configs.append((pr[0], float(pr[1]) if len(pr) > 1 else 1.1, int(pr[2]) if len(pr) > 2 else 112))

    stop = False
    for ci, (backbone, ctx, size) in enumerate(configs):
        if stop or time.time() - START > DEADLINE:
            break
        print(f"=== config {backbone} ctx={ctx} size={size} ===", flush=True)
        t0 = time.time()
        crops_path = extract_crops(train_df, tr_images, ctx, size, box, f"tr{ci}")
        test_path = extract_crops(test_df, te_images, ctx, size, box, f"te{ci}")
        print(f"extracted crops {time.time()-t0:.0f}s", flush=True)
        skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + ci)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n), y, groups)):
            if fold >= MAXFOLDS:
                break
            if time.time() - START > DEADLINE:
                print("DEADLINE reached, stopping", flush=True); stop = True; break
            ft = time.time()
            oof, oof_g, tst = train_fold(crops_path, geom_tr, y, tr_idx, va_idx, test_path, geom_te,
                                         backbone, size, SEED + 100 * ci + fold)
            oof_sum[va_idx] += oof; oof_gsum[va_idx] += oof_g; oof_cnt[va_idx] += 1
            test_sum += tst; test_cnt += 1
            mask = oof_cnt > 0
            P = oof_sum.copy(); P[mask] /= oof_cnt[mask, None]
            bw, sc = tune_weights(P[mask], y[mask])
            lab = np.array(LABELS)[np.argmax((test_sum / max(test_cnt, 1)) * np.array(bw), axis=1)]
            write_sub(test_df, lab, subpath)
            yc = np.array(LABELS)[y[mask]]; pc = np.array(LABELS)[np.argmax(P[mask] * np.array(bw), axis=1)]
            pf = []
            for c in LABELS:
                yt = yc == c; pp = pc == c; tp = (yt & pp).sum()
                pf.append(round(2 * tp / max(2 * tp + (~yt & pp).sum() + (yt & ~pp).sum(), 1), 3))
            print(f"[{backbone} f{fold}] {time.time()-ft:.0f}s cov={int(mask.sum())} OOF_mf1={sc:.4f} "
                  f"perclass={pf} w={tuple(round(x,2) for x in bw)} elapsed={time.time()-START:.0f}s", flush=True)
        for pth in (crops_path, test_path):
            try:
                os.remove(pth)
            except Exception:
                pass
        gc.collect()

    mask = oof_cnt > 0
    P = oof_sum.copy(); P[mask] /= oof_cnt[mask, None]
    if mask.sum() > 0:
        bw, sc = tune_weights(P[mask], y[mask])
        ycls = np.array(LABELS)[y[mask]]
        pred = np.array(LABELS)[np.argmax(P[mask] * np.array(bw), axis=1)]
        print("=== FINAL OOF ===", flush=True)
        print(f"macro_f1={macro_f1(ycls, pred):.4f} weights={tuple(round(x,3) for x in bw)} cov={int(mask.sum())}/{n}", flush=True)
        for c in LABELS:
            yt = ycls == c; pp = pred == c
            tp = (yt & pp).sum(); fp = (~yt & pp).sum(); fn = (yt & ~pp).sum()
            print(f"  class {c}: F1={2*tp/max(2*tp+fp+fn,1):.4f} support={int(yt.sum())} predicted={int(pp.sum())}", flush=True)
        Pg = oof_gsum.copy(); Pg[mask] /= oof_cnt[mask, None]
        bwg, _ = tune_weights(Pg[mask], y[mask])
        predg = np.array(LABELS)[np.argmax(Pg[mask] * np.array(bwg), axis=1)]
        gm = macro_f1(ycls, predg)
        g3 = None
        for c in [3.0]:
            yt = ycls == c; pp = predg == c; tp = (yt & pp).sum()
            g3 = 2 * tp / max(2 * tp + (~yt & pp).sum() + (yt & ~pp).sum(), 1)
        print(f"[CV-contribution] geom-only-head OOF macro={gm:.4f} (class3={g3:.4f}) | "
              f"fused OOF macro={macro_f1(ycls, pred):.4f} | CNN visual lift={macro_f1(ycls, pred)-gm:+.4f}", flush=True)
    else:
        bw = (1.0, 1.0, 1.0)
    if test_cnt == 0:
        lab = np.full(len(test_df), 1.0)
    else:
        lab = np.array(LABELS)[np.argmax((test_sum / test_cnt) * np.array(bw), axis=1)]
    write_sub(test_df, lab, subpath)
    print(f"wrote {subpath} rows={len(lab)} dist={pd.Series(lab).value_counts().sort_index().to_dict()} total={time.time()-START:.0f}s", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)  # avoid Windows DataLoader-worker teardown hang; all outputs already written+flushed
