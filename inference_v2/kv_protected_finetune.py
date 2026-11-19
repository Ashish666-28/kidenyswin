"""Low-storage tumor/cyst specialization with immutable stone/classification paths."""
import copy
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from kv_multitask import SwinMultiTask, DISEASES, MASK_CLASSES

FORMAT = 'kidneyvision_protected_decoder_v1'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def mask_rule(probability, threshold, min_pixels=0):
    mask = np.asarray(probability) >= float(threshold)
    if min_pixels > 0 and mask.any():
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        keep = np.bincount(labels.ravel()) >= int(min_pixels)
        keep[0] = False
        mask = keep[labels]
    return mask


class Specialist(nn.Module):
    def __init__(self, base):
        super().__init__()
        for name in ('lateral', 'fuse', 'stem', 'final'):
            setattr(self, name, copy.deepcopy(getattr(base, name)))
        old = base.head
        self.head = nn.Conv2d(old.in_channels, 2, old.kernel_size,
                              stride=old.stride, padding=old.padding, bias=old.bias is not None)
        with torch.no_grad():
            self.head.weight.copy_(old.weight[1:3])
            if old.bias is not None:
                self.head.bias.copy_(old.bias[1:3])

    def forward(self, x, feats):
        z = self.lateral[3](feats[3])
        for j, i in enumerate((2, 1, 0)):
            z = self.fuse[j](torch.cat((F.interpolate(z, size=feats[i].shape[-2:],
                mode='bilinear', align_corners=False), self.lateral[i](feats[i])), 1))
        z = F.interpolate(z, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.final(torch.cat((z, self.stem(x)), 1)))


class ProtectedSwin(nn.Module):
    """One frozen Swin encoder; old stone/class heads and a new two-channel decoder."""
    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.expert = Specialist(base).requires_grad_(True)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x, task='seg'):
        if task not in ('seg', 'cls', 'both', 'expert'):
            raise ValueError('Invalid task')
        # no_grad, rather than inference_mode: the features feed a trainable decoder.
        with torch.no_grad():
            feats = self.base.encoder(x)
            cls = self.base.cls_head(feats[-1].mean((2, 3))) if task in ('cls', 'both') else None
            stone = self.base._seg_from_features(x, feats)[:, :1] if task in ('seg', 'both') else None
        if task == 'cls':
            return cls
        tc = self.expert(x, feats)
        if task == 'expert':
            return tc
        seg = torch.cat((stone, tc), 1)
        return {'classification_logits': cls, 'segmentation_logits': seg} if task == 'both' else seg


def load_parent(path, device):
    state = torch.load(str(path), map_location='cpu', weights_only=True)
    if state.get('format') == FORMAT:
        return load_protected(path, device)
    if list(state.get('classes', [])) != list(DISEASES) or list(state.get('mask_classes', [])) != list(MASK_CLASSES):
        raise ValueError('Requires your four-class + three-mask multitask checkpoint, with class metadata.')
    base = SwinMultiTask(state['encoder_config'])
    base.load_state_dict(state['model'], strict=True)
    net = ProtectedSwin(base).to(device).eval()
    net.segmentation_thresholds = list(map(float, state.get('segmentation_thresholds', [.5] * 3)))
    return net, state


def load_protected(path, device='cpu'):
    state = torch.load(str(path), map_location='cpu', weights_only=True)
    if state.get('format') != FORMAT:
        raise ValueError('Use this loader only with the protected-decoder export.')
    net = ProtectedSwin(SwinMultiTask(state['encoder_config']))
    net.load_state_dict(state['model'], strict=True)
    net.segmentation_thresholds = list(map(float, state['segmentation_thresholds']))
    if tensor_hash(net.base.state_dict()) != state['frozen_base_sha256']:
        raise ValueError('Frozen branch checksum mismatch.')
    return net.to(device).eval(), state


def resolve_manifest_paths(frame, root):
    """Relocate paths by project-relative suffix; never drop rows or choose by basename."""
    frame = frame.copy()
    marker = '/' + Path(root).name + '/'
    def resolve(value):
        text = str(value).replace('\\', '/')
        if marker in text:
            return str(Path(root) / text.split(marker, 1)[1])
        if not Path(text).is_absolute():
            return str(Path(root) / text)
        return text
    frame['path'] = frame.path.map(resolve)
    return frame


def training_plan(frame, seed, minimum_per_class=256):
    """Every positive training patient for EACH disease, then balanced extra draws."""
    rng = np.random.default_rng(seed)
    plan = []
    for focus, name in ((1, 'tumor'), (2, 'cyst')):
        groups = [g for _, g in frame[frame['has_' + name] == 1].groupby('group', sort=True)]
        if not groups:
            raise ValueError('No training positives for ' + name)
        for g in groups:
            plan.append((int(rng.choice(g.index)), focus))
        for _ in range(max(0, int(minimum_per_class) - len(groups))):
            g = groups[int(rng.integers(len(groups)))]
            plan.append((int(rng.choice(g.index)), focus))
    rng.shuffle(plan)
    return plan


class PositiveCache:
    """Bounded positive-slice SSD cache; training-only unless evaluation explicitly opts in."""
    def __init__(self, directory, maximum_bytes, reserve_bytes=2 * 1024**3,
                 allowed_splits=('train',)):
        self.allowed_splits = frozenset(allowed_splits)
        if not self.allowed_splits or not self.allowed_splits.issubset({'train', 'val', 'test'}):
            raise ValueError('Cache requires explicit valid dataset splits.')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit = max(0, int(maximum_bytes))
        self.reserve = int(reserve_bytes)
        self.items = {}
        self.hits = self.copies = 0
        # This dedicated cache belongs only to this experiment; no shared data is removed.
        for p in self.directory.glob('*.npz'):
            self.items[p.name] = (p, p.stat().st_size, p.stat().st_mtime)
        self.evict_to(self.limit)

    def evict_to(self, target):
        while self.items and sum(x[1] for x in self.items.values()) > max(0, target):
            key = min(self.items, key=lambda k: self.items[k][2])
            self.items.pop(key)[0].unlink(missing_ok=True)

    def get(self, row):
        if str(row.source) != 'kits' or str(row.split) not in self.allowed_splits or not (int(row.has_tumor) or int(row.has_cyst)):
            raise ValueError('Cache accepts only KiTS tumor/cyst-positive slices from its explicitly allowed splits.')
        source = Path(row.path)
        key = hashlib.sha256(str(source).encode()).hexdigest() + '.npz'
        dest = self.directory / key
        if key in self.items and dest.is_file():
            p, size, _ = self.items[key]
            self.items[key] = (p, size, time.time())
            self.hits += 1
            return dest
        try:
            size = source.stat().st_size
        except OSError as exc:
            raise OSError('Cannot read shared prepared slice; no samples were silently skipped: ' + str(source)) from exc
        if size > self.limit:
            return source
        self.evict_to(self.limit - size)
        if shutil.disk_usage(self.directory).free < size + self.reserve:
            return source  # Stream from shared Drive when local SSD cannot accommodate it.
        temp = dest.with_suffix('.part')
        try:
            shutil.copyfile(source, temp)
            if temp.stat().st_size != size:
                raise OSError('Incomplete local cache copy')
            with np.load(temp, allow_pickle=False) as d:
                y = d['y']
                if y.ndim != 3 or y.shape[0] != 3 or not y[1:3].any():
                    raise ValueError('Manifest-positive file has no tumor/cyst mask: ' + str(source))
            os.replace(temp, dest)
        finally:
            temp.unlink(missing_ok=True)
        self.items[key] = (dest, size, time.time())
        self.copies += 1
        return dest


def lesion_crop(x, y, tile, focus, rng, background=False):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.shape != (3, *x.shape) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid prepared x/y arrays')
    points = np.argwhere(y[focus] > 0)
    if not len(points):
        raise ValueError('A manifest-positive slice has no requested lesion pixels.')
    ph, pw = max(0, tile - x.shape[0]), max(0, tile - x.shape[1])
    x = np.pad(x, ((0, ph), (0, pw)))
    y = np.pad(y, ((0, 0), (0, ph), (0, pw)), constant_values=-1)
    h, w = x.shape
    if background:
        for _ in range(16):
            top, left = int(rng.integers(h - tile + 1)), int(rng.integers(w - tile + 1))
            yy = y[:, top:top + tile, left:left + tile]
            if not (yy[1:3] > 0).any():
                return x[top:top + tile, left:left + tile], yy
    cy, cx = points[int(rng.integers(len(points)))]
    top = int(np.clip(cy - int(rng.integers(tile // 4, 3 * tile // 4)), 0, h - tile))
    left = int(np.clip(cx - int(rng.integers(tile // 4, 3 * tile // 4)), 0, w - tile))
    xx, yy = x[top:top + tile, left:left + tile], y[:, top:top + tile, left:left + tile]
    if not (yy[focus] > 0).any():
        raise AssertionError('Target disappeared in positive crop')
    return xx, yy


class FocusDataset(Dataset):
    def __init__(self, frame, plan, cache, tile, seed):
        self.frame, self.plan, self.cache, self.tile, self.seed = frame, plan, cache, tile, seed
    def __len__(self):
        return len(self.plan)
    def __getitem__(self, index):
        idx, focus = self.plan[index]
        row = self.frame.loc[idx]
        rng = np.random.default_rng(self.seed + index)
        path = self.cache.get(row)
        with np.load(path, allow_pickle=False) as d:
            x, y = lesion_crop(d['x'], d['y'], self.tile, focus, rng, background=(index % 4 == 3))
        if index % 2:
            x, y = x[:, ::-1], y[:, :, ::-1]
        known = np.asarray([row.known_tumor, row.known_cyst], np.float32)
        if not np.isin(known, [0, 1]).all() or known[focus - 1] != 1:
            raise ValueError('Positive class is not annotated')
        return torch.from_numpy(x.copy())[None], torch.from_numpy(y[1:3].copy()), torch.from_numpy(known)


def focused_loss(logits, target, known):
    z = logits.float()
    valid = (target >= 0).float()
    y = target.clamp(0, 1).float()
    positive, negative = y * valid, (1 - y) * valid
    area = positive.sum((2, 3))
    pos = (F.softplus(-z) * positive).sum((2, 3)) / area.clamp_min(1)
    neg = (F.softplus(z) * negative).sum((2, 3)) / negative.sum((2, 3)).clamp_min(1)
    bce = torch.where(area > 0, .5 * pos + .5 * neg, neg)
    p = z.sigmoid() * valid
    dice = 1 - (2 * (p * y).sum((2, 3)) + 1) / (p.sum((2, 3)) + area + 1)
    loss = .5 * bce + .5 * dice * (area > 0)
    weight = known.float() * (valid.sum((2, 3)) > 0)
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def validation_screen(frame, seed=42, cases_per_class=8):
    val = frame[(frame.source == 'kits') & (frame.split == 'val')].copy()
    parts = []
    for name in ('tumor', 'cyst'):
        pool = val[val['has_' + name] == 1]
        if pool.empty:
            raise ValueError('Validation lacks ' + name + ' positives')
        parts.append(pool.sample(frac=1, random_state=seed).drop_duplicates('group').head(cases_per_class))
    neg = val[(val.has_tumor == 0) & (val.has_cyst == 0)]
    if neg.empty:
        raise ValueError('Validation needs tumor/cyst-negative slices to measure false positives.')
    parts.append(neg.sample(frac=1, random_state=seed).drop_duplicates('group').head(cases_per_class))
    return pd.concat(parts).drop_duplicates('id').reset_index(drop=True)


@torch.inference_mode()
def evaluate(net, rows, meta, device, infer_batch=4):
    from kv_infer import predict_slice
    net.eval()
    thresholds = meta.get('segmentation_thresholds', [.5] * 3)
    post = meta.get('mask_postprocess', {})
    totals = np.zeros((2, 3), np.float64)
    patients = {}
    for step, row in enumerate(rows.itertuples(), 1):
        # Validation is streamed directly, including negative slices; it is NEVER cached.
        with np.load(row.path, allow_pickle=False) as d:
            x, gt = d['x'].astype(np.float32), d['y'][1:3] > 0
        prob = predict_slice(net, x, tile=int(meta.get('tile_size', 256)), device=device,
                             infer_batch=infer_batch, overlap=.25, tta=False)[1:3]
        case = patients.setdefault(str(row.group), np.zeros((2, 3), np.float64))
        for i, name in enumerate(('tumor', 'cyst')):
            mask = mask_rule(prob[i], thresholds[i + 1], int(post.get(name, {}).get('min_pixels', 0)))
            counts = np.asarray([(mask & gt[i]).sum(), mask.sum(), gt[i].sum()], np.float64)
            totals[i] += counts
            case[i] += counts
        if step % 8 == 0 or step == len(rows):
            print(f'Validation {step}/{len(rows)} slices (streamed, no cache)', flush=True)
    result = {}
    for i, name in enumerate(('tumor', 'cyst')):
        tp, pred, actual = totals[i]
        pp = [2 * a[i, 0] / max(a[i, 1] + a[i, 2], 1) for a in patients.values() if a[i, 2] > 0]
        result[name] = {'dice': float(2 * tp / max(pred + actual, 1)),
            'positive_patient_mean_dice': float(np.mean(pp)) if pp else 0.,
            'precision': float(tp / max(pred, 1)), 'recall': float(tp / max(actual, 1))}
    return result


def improves(candidate, best, minimum_gain=.0005):
    values = []
    for name in ('tumor', 'cyst'):
        for metric in ('dice', 'positive_patient_mean_dice'):
            value, old = candidate[name][metric], best[name][metric]
            if not math.isfinite(value) or value < old - 1e-8:
                return False
            values.append(value - old)
    return sum(values) / len(values) >= minimum_gain


@torch.inference_mode()
def assert_protected(net, base_sha, tile, device):
    if any(p.requires_grad for p in net.base.parameters()):
        raise RuntimeError('Frozen parameter became trainable')
    if tensor_hash(net.base.state_dict()) != base_sha:
        raise RuntimeError('Frozen stone/classification branch changed; refusing checkpoint.')
    net.eval()
    generator = torch.Generator().manual_seed(123)
    x = torch.rand((1, 1, tile, tile), generator=generator).to(device)
    with torch.autocast('cuda', dtype=torch.float16, enabled=str(device).startswith('cuda')):
        old = net.base(x, task='both')
        new = net(x, task='both')
    for a, b in ((old['classification_logits'], new['classification_logits']),
                 (old['segmentation_logits'][:, 0], new['segmentation_logits'][:, 0])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    print('Protection passed: frozen tensor checksum + identical stone/class logits on audit input.', flush=True)


def fit(source_path, frame, output_dir, settings):
    device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError('Select a T4 GPU runtime.')
    source_path, output_dir = Path(source_path), Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError('Set SOURCE_MODEL_PATH to your actual shared model.pth: ' + str(source_path))
    parent_sha = file_hash(source_path)
    net, parent = load_parent(source_path, device)
    tile = int(parent.get('tile_size', 256))
    if tile < 64 or tile % 32:
        raise ValueError('Unsupported saved tile size')
    meta = {k: parent[k] for k in ('encoder_config', 'classes', 'mask_classes', 'tile_size',
             'segmentation_thresholds', 'mask_postprocess', 'epoch') if k in parent}
    meta.setdefault('segmentation_thresholds', [.5] * 3)
    meta.setdefault('tile_size', tile)
    base_sha = tensor_hash(net.base.state_dict())
    signature_settings = {k: settings[k] for k in ('SEED', 'BATCH', 'LR', 'MINIMUM_PER_CLASS', 'SCREEN_CASES')}
    screen = validation_screen(frame, settings['SEED'], settings['SCREEN_CASES'])
    train = frame[(frame.source == 'kits') & (frame.split == 'train') &
                  ((frame.has_tumor == 1) | (frame.has_cyst == 1))].copy().reset_index(drop=True)
    if set(train.group) & set(screen.group):
        raise ValueError('Training and validation patients overlap')
    signature = hashlib.sha256(json.dumps({'parent': parent_sha, 'settings': signature_settings,
        'train': train[['id', 'group', 'has_tumor', 'has_cyst']].astype(str).values.tolist(),
        'screen': screen[['id', 'group']].astype(str).values.tolist()}, sort_keys=True).encode()).hexdigest()
    run_dir = output_dir / signature[:12]
    run_dir.mkdir(parents=True, exist_ok=True)
    last, best_path = run_dir / 'last_finetune.pth', run_dir / 'best_expert.pth'
    optimizer = torch.optim.AdamW(net.expert.parameters(), lr=settings['LR'], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    completed = next_step = bad = 0
    history = []
    best_metrics = None
    if last.is_file():
        saved = torch.load(last, map_location='cpu', weights_only=True)
        if saved['signature'] != signature or saved['frozen_base_sha256'] != base_sha:
            raise ValueError('Resume fingerprint mismatch')
        net.expert.load_state_dict(saved['expert'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        completed, next_step = int(saved['completed_rounds']), int(saved['next_step'])
        history, best_metrics, bad = saved['history'], saved['best_metrics'], int(saved['bad_rounds'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    assert_protected(net, base_sha, tile, device)
    print('Original saved epoch:', parent.get('epoch', 'not recorded'), '| completed specialist rounds:', completed,
          '| next batch:', next_step + 1, flush=True)
    print('Eligible positive training patients:', train.group.nunique(), '| positive slices:', len(train),
          '| no healthy-only cases are claimed as trained in this phase.', flush=True)
    if best_metrics is None:
        best_metrics = evaluate(net, screen, meta, device, settings['INFER_BATCH'])
        atomic_save({'expert': net.expert.state_dict(), 'metrics': best_metrics, 'round': 0,
                     'signature': signature, 'frozen_base_sha256': base_sha}, best_path)
        print('Starting validation screen:', best_metrics, flush=True)
    elif not best_path.is_file():
        raise FileNotFoundError('Best expert is missing; restore it before resuming.')
    cache = PositiveCache(Path(settings['CACHE_DIR']) / signature[:12], settings['CACHE_BYTES'])
    trainable = {id(p) for g in optimizer.param_groups for p in g['params']}
    if trainable & {id(p) for p in net.base.parameters()}:
        raise RuntimeError('Optimizer includes protected parameters')
    started = time.monotonic()
    deadline = started + 60 * settings['MINUTES']
    def save_progress():
        assert tensor_hash(net.base.state_dict()) == base_sha
        atomic_save({'expert': net.expert.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'signature': signature, 'frozen_base_sha256': base_sha,
            'source_sha256': parent_sha, 'source_path': str(source_path), 'completed_rounds': completed,
            'next_step': next_step, 'best_metrics': best_metrics, 'bad_rounds': bad, 'history': history,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, last)
    # Save a durable baseline even if this session cannot finish a complete round.
    save_progress()
    for round_number in range(completed + 1, settings['MAX_ROUNDS'] + 1):
        if bad >= settings['PATIENCE'] or time.monotonic() >= deadline:
            break
        plan = training_plan(train, settings['SEED'] + round_number, settings['MINIMUM_PER_CLASS'])
        ds = FocusDataset(train, plan, cache, tile, settings['SEED'] + 100000 * round_number)
        # One process owns the bounded cache. No worker races or Drive subprocess stalls.
        first_item = next_step * settings['BATCH']
        indices = range(min(first_item, len(ds)), len(ds))
        loader = DataLoader(torch.utils.data.Subset(ds, indices), batch_size=settings['BATCH'],
                            shuffle=False, num_workers=0, pin_memory=True)
        total_steps = math.ceil(len(ds) / settings['BATCH'])
        net.train()
        try:
            for x, y, known in loader:
                if time.monotonic() >= deadline:
                    save_progress()
                    print('Time budget reached. Partial round saved; next run resumes its next batch.', flush=True)
                    return run_dir
                x, y, known = [a.to(device, non_blocking=True) for a in (x, y, known)]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = focused_loss(net(x, task='expert'), y, known)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss; last valid checkpoint remains available')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.expert.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                next_step += 1
                if next_step == 1 or next_step % 25 == 0 or next_step == total_steps:
                    print(f'Round {round_number}, batch {next_step}/{total_steps}, loss={loss.item():.4f}', flush=True)
                if next_step % settings['SAVE_EVERY'] == 0:
                    save_progress()
            # Persist finished batches before potentially slow streamed validation.
            save_progress()
            metrics = evaluate(net, screen, meta, device, settings['INFER_BATCH'])
            assert_protected(net, base_sha, tile, device)
            accepted = improves(metrics, best_metrics)
            if accepted:
                best_metrics, bad = metrics, 0
                atomic_save({'expert': net.expert.state_dict(), 'metrics': metrics, 'round': round_number,
                    'signature': signature, 'frozen_base_sha256': base_sha}, best_path)
            else:
                bad += 1
            history.append({'round': round_number, 'metrics': metrics, 'accepted': accepted})
            completed, next_step = round_number, 0
            save_progress()
            print('Validation:', metrics, '| accepted:', accepted, '| best:', best_metrics, flush=True)
        except (KeyboardInterrupt, torch.cuda.OutOfMemoryError):
            optimizer.zero_grad(set_to_none=True)
            save_progress()
            print('Saved recovery checkpoint. Re-run to continue. For OOM, restart before changing batch settings.')
            raise
    print('Saved run:', run_dir, '| best screen:', best_metrics, '| cache GB:',
          round(sum(v[1] for v in cache.items.values()) / 1024**3, 3), flush=True)
    return run_dir


def export_best(source_path, run_dir):
    source_path, run_dir = Path(source_path), Path(run_dir)
    net, parent = load_parent(source_path, 'cpu')
    best = torch.load(run_dir / 'best_expert.pth', map_location='cpu', weights_only=True)
    progress = torch.load(run_dir / 'last_finetune.pth', map_location='cpu', weights_only=True)
    if progress['source_sha256'] != file_hash(source_path) or best['signature'] != progress['signature']:
        raise ValueError('Export source/run mismatch')
    net.expert.load_state_dict(best['expert'], strict=True)
    if tensor_hash(net.base.state_dict()) != best['frozen_base_sha256']:
        raise ValueError('Protected branch changed')
    out = run_dir / 'export'
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: parent[k] for k in ('encoder_config', 'classes', 'mask_classes', 'tile_size',
        'segmentation_thresholds', 'mask_postprocess', 'epoch') if k in parent}
    meta.setdefault('segmentation_thresholds', [.5] * 3)
    meta.setdefault('tile_size', 256)
    bundle = dict(meta, format=FORMAT, model=net.state_dict(), frozen_base_sha256=best['frozen_base_sha256'],
                  parent_sha256=progress['source_sha256'], specialist_round=best['round'],
                  validation_screen=best['metrics'], research_only=True)
    atomic_save(bundle, out / 'model.pth')
    for name in ('kv_protected_finetune.py', 'kv_multitask.py', 'kv_model.py', 'kv_data.py', 'kv_infer.py', 'kv_joint_infer.py'):
        module = __import__(name[:-3])
        shutil.copyfile(module.__file__, out / name)
    (out / 'requirements.txt').write_text('torch>=2.3\nnumpy\npandas\nscipy\nPillow\nmatplotlib\nnibabel\npydicom\nrequests\n')
    (out / 'README.txt').write_text(
        'Use kv_protected_finetune.load_protected("model.pth", device). Old loaders do not support this expanded architecture.\n'
        'Use kv_protected_finetune.predict_image for matching thresholds and postprocessing.\n'
        'Classification order: normal, stone, tumor, cyst. Masks: stone, tumor, cyst.\n'
        'Background means no predicted lesion; it is not a separately validated normal-tissue class.\n'
        'The stone and classification branches retain their source checkpoint weights and preprocessing.\n'
        'The first saved best can be baseline round 0 if no candidate improves both diseases.\n'
        'Resume training using the shared source checkpoint and last_finetune.pth, not this compact export.\n'
        'Small validation-screen metrics are development results, not final test accuracy.\n')
    # Test strict load of the actual saved export, without allocating the GPU twice.
    check, _ = load_protected(out / 'model.pth', 'cpu')
    del check
    print('Exported:', out / 'model.pth', '| selected specialist round:', best['round'], flush=True)
    return out


@torch.inference_mode()
def predict_image(model_path, image_path, outdir, device='cpu'):
    from kv_joint_infer import read_2d, predict_classification
    from kv_infer import predict_slice, overlay, components_2d
    from PIL import Image
    net, meta = load_protected(model_path, device)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x, spacing, provenance = read_2d(image_path)
    tile = int(meta['tile_size'])
    probs = predict_slice(net, x, tile=tile, overlap=.25, device=device, tta=False, infer_batch=4)
    post = meta.get('mask_postprocess', {})
    masks = np.stack([mask_rule(probs[i], meta['segmentation_thresholds'][i],
        int(post.get(name, {}).get('min_pixels', 0))) for i, name in enumerate(MASK_CLASSES)])
    cls = predict_classification(net, x, tile=tile, device=device)
    overlay(x, masks, outdir / 'highlighted_ct.png')
    # Stone has display priority so a new tumor/cyst mask cannot erase a stone label.
    label = np.zeros(x.shape, np.uint8)
    label[masks[2]] = 3
    label[masks[1]] = 2
    label[masks[0]] = 1
    Image.fromarray(label).save(outdir / 'segmentation_labels.png')
    np.savez_compressed(outdir / 'independent_masks.npz', masks=masks, probabilities=probs)
    rows = components_2d(masks, spacing=spacing, provenance=provenance or 'No verified spacing; pixels only')
    pd.DataFrame(rows).to_csv(outdir / 'lesions.csv', index=False)
    result = {'class_probabilities': cls, 'class': max(cls, key=cls.get), 'lesions': rows,
        'overlay': str(outdir / 'highlighted_ct.png'), 'overlap_pixels': int((masks.sum(0) > 1).sum()),
        'note': 'Display labels: 0=no predicted lesion, 1=stone, 2=tumor, 3=cyst. Independent masks are primary; overlaps are retained there.'}
    (outdir / 'prediction.json').write_text(json.dumps(result, indent=2))
    return result
