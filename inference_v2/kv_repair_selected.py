from pathlib import Path

import gc

import hashlib

import json

import os

import shutil

import tempfile

import time

import numpy as np

import pandas as pd

def _repair_json(path):
    with Path(path).open() as stream:
        return json.load(stream)

def _repair_readable(path, delays=(0, 1, 3)):
    """Check actual open/read, rather than relying only on a stale Drive directory listing."""
    last = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            with Path(path).open('rb') as stream:
                if not stream.read(1):
                    raise ValueError('Prepared file is empty: ' + str(path))
            return True
        except FileNotFoundError as exc:
            last = exc
        except OSError as exc:
            # Errno 5 is a mounted-Drive failure, not evidence that a case is absent.
            if exc.errno == 5:
                raise RuntimeError('Google Drive returned I/O error 5. Reconnect Drive before repairing; '
                                   'no case was excluded or declared missing.') from exc
            raise
    return False

def _repair_audit(frame):
    missing = []
    grouped = frame.groupby(frame.path.map(lambda p: str(Path(p).parent)), sort=True)
    for number, (folder, rows) in enumerate(grouped, 1):
        try:
            names = {entry.name for entry in os.scandir(folder)}
        except FileNotFoundError:
            names = set()
        except OSError as exc:
            raise RuntimeError('Shared folder could not be listed; reconnect Drive: ' + str(folder)) from exc
        for _, row in rows.iterrows():
            if Path(row.path).name not in names and not _repair_readable(row.path):
                missing.append(row.to_dict())
        if number % 50 == 0 or number == len(grouped):
            print(f'Selected-slice audit: {number}/{len(grouped)} case folders; '
                  f'{len(missing)} missing requested paths found', flush=True)
    return pd.DataFrame(missing, columns=frame.columns)

def _repair_arrays(image, labels, row):
    z = int(row['slice'])
    if z < 0 or z >= image.shape[2] or Path(row['path']).name != f'{z:04d}.npz':
        raise ValueError('Slice number/filename mismatch; refusing to substitute a different CT section.')
    from kv_data import window_hu
    x = window_hu(image[:, :, z].T).astype(np.float16)
    mask = labels[:, :, z].T
    y = np.zeros((3, *x.shape), np.uint8)
    y[1] = mask == 2
    y[2] = mask == 3
    if int(y[1].any()) != int(row['has_tumor']) or int(y[2].any()) != int(row['has_cyst']):
        raise ValueError('Original annotation and saved manifest disagree at ' + str(row['path']))
    return x, y

def _repair_selection(root):
    path = root/'raw'/'selected_cases.json'
    if path.is_file():
        result = _repair_json(path)
    else:
        policy = root/'processed'/'policy.json'
        result = _repair_json(policy).get('selection', {}) if policy.is_file() else {}
    for key in ('cases', 'kits_revision', 'hf_revision'):
        if not result.get(key):
            raise FileNotFoundError('The saved source revision is unavailable. Restore raw/selected_cases.json '
                                    'or processed/policy.json; no unpinned dataset will be substituted.')
    return result

def _repair_raw_paths(root, case, geometry):
    """Only explicit same-case locations; no whole-Drive search or basename-only pairing."""
    pair = [root/'raw'/'kits'/case/'imaging.nii.gz',
            root/'raw'/'kits'/case/'segmentation.nii.gz']
    for index, key in enumerate(('image_path', 'mask_path')):
        text = str(geometry.get(key, ''))
        if not text:
            continue
        marker = '/' + root.name + '/'
        candidate = root/text.split(marker, 1)[1] if marker in text else Path(text)
        if candidate.parent.name == case and candidate.is_file():
            pair[index] = candidate
    return pair

def repair_selected(root, frame, allow_download=True):
    root = Path(root)
    needed = {'id', 'path', 'group', 'split', 'source', 'slice', 'has_tumor', 'has_cyst'}
    if not needed.issubset(frame.columns):
        raise ValueError('Repair requires original manifest columns: ' + str(needed-set(frame.columns)))
    positive = frame.copy()
    if positive.empty:
        raise ValueError('No positive training rows; run setup cell 3 first.')
    # Require the unmodified original destination layout so new data cannot land in a local cache.
    for _, row in positive.iterrows():
        expected = root/'processed'/'kits'/str(row.group)/f'{int(row["slice"]):04d}.npz'
        if Path(row.path) != expected:
            raise ValueError('Unexpected prepared path. Correct the shared ROOT/path mapping first: ' + str(row.path))
    missing = _repair_audit(positive)
    if missing.empty:
        print(f'Selected-slice audit passed: {len(positive)} requested rows; no reconstruction.',flush=True)
        return {'repaired_slices':0,'checked_rows':len(positive)}
    print(f'Restoring {len(missing)} exact requested slices from {missing.group.nunique()} case(s). '
          'Model weights, manifest rows, and splits remain unchanged.', flush=True)
    selection = _repair_selection(root)
    from kv_data import canonical_pair, download, fetch_json
    tree = None
    tree_path = root/'raw'/'kits_tree.json'
    if tree_path.is_file():
        tree = _repair_json(tree_path)
    repaired = 0
    for case, rows in missing.groupby('group', sort=True):
        case = str(case)
        if case not in selection['cases']:
            raise ValueError('Case was not in the original selected cohort: ' + case)
        folder = root/'processed'/'kits'/case
        geometry_path = folder/'geometry.json'
        geometry = _repair_json(geometry_path) if geometry_path.is_file() else {}
        image_path, mask_path = _repair_raw_paths(root, case, geometry)
        with tempfile.TemporaryDirectory(prefix='kv_missing_raw_', dir='/content') as temp:
            temp = Path(temp)
            if not image_path.is_file() or not mask_path.is_file():
                if not allow_download:
                    raise FileNotFoundError('Original raw CT/mask is absent for ' + case + '. Enable '
                        'ALLOW_TEMPORARY_RAW_DOWNLOAD or restore those source files. No patient was skipped.')
                # Leave space for a compressed CT and decompression/processing. Only one case is handled at a time.
                if shutil.disk_usage(temp).free < 2 * 1024**3:
                    raise RuntimeError('Need at least 2 GiB free temporary Colab disk for one raw-case repair. '
                                       'No new raw volume will be saved to Google Drive.')
                if tree is None:
                    tree = fetch_json('https://api.github.com/repos/neheller/kits23/git/trees/' +
                                      selection['kits_revision'] + '?recursive=1', temp/'kits_tree.json')
                if tree.get('truncated'):
                    raise ValueError('Pinned KiTS source tree was truncated')
                mask_record = next((item for item in tree['tree']
                                    if item['path'] == f'dataset/{case}/segmentation.nii.gz'), None)
                if mask_record is None:
                    raise ValueError('Case mask not found in original pinned annotation tree: ' + case)
                if not mask_path.is_file():
                    mask_path = temp/'segmentation.nii.gz'
                    download('https://raw.githubusercontent.com/neheller/kits23/' + selection['kits_revision'] +
                             '/' + mask_record['path'], mask_path,
                             expected_bytes=mask_record['size'], git_blob=mask_record['sha'])
                if not image_path.is_file():
                    image_path = temp/'imaging.nii.gz'
                    print('Downloading one temporary CT volume to restore', case, flush=True)
                    download('https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/' +
                             selection['hf_revision'] + f'/images/{case}.nii.gz', image_path)
            image, labels, actual_geometry = canonical_pair(str(image_path), str(mask_path), unit_override=None)
            for key in ('canonical_shape', 'original_shape'):
                if key in geometry and list(geometry[key]) != list(actual_geometry[key]):
                    raise ValueError('Saved and raw image shapes disagree: ' + case)
            if 'canonical_affine' in geometry and not np.allclose(
                    geometry['canonical_affine'], actual_geometry['canonical_affine'], atol=1e-4):
                raise ValueError('Saved and raw orientation/geometry disagree: ' + case)
            # Verify this exact preparation against existing same-case slices when any remain.
            case_rows = frame[(frame.source == 'kits') & (frame.group.astype(str) == case)]
            checked = 0
            for _, check_row in case_rows.iterrows():
                if Path(check_row.path).is_file():
                    xx, yy = _repair_arrays(image, labels, check_row)
                    with np.load(check_row.path, allow_pickle=False) as d:
                        if not np.array_equal(d['x'], xx) or not np.array_equal(d['y'], yy):
                            raise ValueError('Reconstructed values differ from the existing same-case preparation: ' + str(check_row.path))
                    checked += 1
                    if checked == 2:
                        break
            print(case, '| matching existing reference slices:', checked, flush=True)
            folder.mkdir(parents=True, exist_ok=True)
            for _, row in rows.iterrows():
                dest = Path(row.path)
                # A delayed Drive file may have appeared while the raw case was fetched.
                if _repair_readable(dest, delays=(0,)):
                    continue
                x, y = _repair_arrays(image, labels, row)
                temp_npz = dest.with_name(dest.name + '.repair.part')
                try:
                    with temp_npz.open('wb') as stream:
                        np.savez_compressed(stream, x=x, y=y)
                    with np.load(temp_npz, allow_pickle=False) as d:
                        if not np.array_equal(d['x'], x) or not np.array_equal(d['y'], y):
                            raise OSError('Repair write verification failed: ' + str(dest))
                    os.replace(temp_npz, dest)
                finally:
                    temp_npz.unlink(missing_ok=True)
                if not _repair_readable(dest):
                    raise OSError('Drive did not expose the repaired file: ' + str(dest))
                repaired += 1
            del image, labels
            gc.collect()
        print(case, 'selected-slice repair finished; temporary raw download removed.', flush=True)
    unresolved = _repair_audit(positive)
    if not unresolved.empty:
        raise FileNotFoundError('Still unreachable; no rows were removed: ' + str(unresolved.path.head(10).to_list()))
    print(f'REPAIR COMPLETE: {repaired} requested slices restored. Requested slices are now available. '
          'The original saved epoch and fine-tuning resume checkpoint were not changed.', flush=True)
    return {'repaired_slices': repaired, 'checked_rows': len(positive)}