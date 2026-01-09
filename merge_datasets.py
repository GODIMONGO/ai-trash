#!/usr/bin/env python3
"""
merge_datasets.py

Dry-run by default: inspects `new/` dataset and shows what would be copied and how labels
would be remapped into the main dataset. Use --apply to perform the copy and to update
`data.yaml` (a backup `data.yaml.bak` will be created).

Usage:
  python merge_datasets.py        # dry-run
  python merge_datasets.py --apply

The script assumes the workspace root is the current directory and expects folders:
  - train/images, train/labels
  - valid/images, valid/labels (optional)
  - test/images, test/labels (optional)
  - new/train/images, new/train/labels, etc.

It will merge classes by name (deduplicating). Labels from `new/` are remapped to the
merged class indices. Filenames from `new/` are prefixed with `new_` to avoid collisions.
"""

import argparse
import os
import shutil
import sys
from collections import defaultdict

try:
    import yaml
except Exception:
    yaml = None


ROOT = os.path.abspath(os.path.dirname(__file__))


def load_yaml(path):
    if yaml:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    # minimal fallback parser for simple YAML used here
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    for line in lines:
        if line.startswith('nc:'):
            d['nc'] = int(line.split(':',1)[1].strip())
        elif line.startswith('names:'):
            # assume names: ['a','b'] on one line
            val = line.split(':',1)[1].strip()
            if val.startswith('[') and val.endswith(']'):
                items = val[1:-1].strip()
                if items:
                    d['names'] = [s.strip().strip("'\"") for s in items.split(',')]
                else:
                    d['names'] = []
    return d


def find_subsets(new_root):
    # find train, valid, test directories under new_root
    subs = {}
    for sub in ('train', 'valid', 'test'):
        p = os.path.join(new_root, sub)
        if os.path.isdir(p):
            subs[sub] = p
    return subs


def list_files(images_dir, labels_dir):
    imgs = []
    if os.path.isdir(images_dir):
        for fn in os.listdir(images_dir):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                imgs.append(fn)
    return imgs


def ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def main(dry_run=True):
    root = ROOT
    main_data_yaml = os.path.join(root, 'data.yaml')
    new_data_yaml = os.path.join(root, 'new', 'data.yaml')

    if not os.path.isfile(main_data_yaml):
        print('ERROR: main data.yaml not found at', main_data_yaml)
        sys.exit(2)

    if not os.path.isdir(os.path.join(root, 'new')):
        print('ERROR: folder `new/` not found in', root)
        sys.exit(2)

    main_cfg = load_yaml(main_data_yaml)
    new_cfg = load_yaml(new_data_yaml) if os.path.isfile(new_data_yaml) else {}

    main_names = main_cfg.get('names') or []
    new_names = new_cfg.get('names') or []

    merged_names = list(main_names)
    for n in new_names:
        if n not in merged_names:
            merged_names.append(n)

    print('Main classes ({}):'.format(len(main_names)), main_names)
    print('New classes ({}):'.format(len(new_names)), new_names)
    print('Merged classes ({}):'.format(len(merged_names)), merged_names)

    # mapping for new dataset indices -> merged indices
    new_to_merged = {i: merged_names.index(name) for i, name in enumerate(new_names)}

    # Prepare summary counters
    summary = defaultdict(int)
    planned_ops = []

    # For each subset present under new/, merge into corresponding main folder
    for subset in ('train', 'valid', 'test'):
        new_images_dir = os.path.join(root, 'new', subset, 'images')
        new_labels_dir = os.path.join(root, 'new', subset, 'labels')
        main_images_dir = os.path.join(root, subset, 'images')
        main_labels_dir = os.path.join(root, subset, 'labels')

        if not os.path.isdir(new_images_dir):
            continue

        ensure_dir(main_images_dir)
        ensure_dir(main_labels_dir)

        for img_name in list_files(new_images_dir, new_labels_dir):
            src_img = os.path.join(new_images_dir, img_name)
            stem, ext = os.path.splitext(img_name)
            dst_img_name = f'new_{img_name}'
            dst_img = os.path.join(main_images_dir, dst_img_name)

            # ensure unique name
            k = 1
            while os.path.exists(dst_img):
                dst_img_name = f'new_{stem}_{k}{ext}'
                dst_img = os.path.join(main_images_dir, dst_img_name)
                k += 1

            # label handling
            src_lbl = os.path.join(new_labels_dir, stem + '.txt')
            dst_lbl = os.path.join(main_labels_dir, os.path.splitext(dst_img_name)[0] + '.txt')

            if os.path.exists(src_lbl):
                # read and remap
                with open(src_lbl, 'r', encoding='utf-8') as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                remapped = []
                for ln in lines:
                    parts = ln.split()
                    if len(parts) < 5:
                        print(f'Warning: malformed label {src_lbl}: {ln}')
                        continue
                    try:
                        cid = int(parts[0])
                    except Exception:
                        print(f'Warning: cannot parse class id in {src_lbl}: {ln}')
                        continue
                    if cid in new_to_merged:
                        new_cid = new_to_merged[cid]
                    else:
                        # this can happen if new label references class not in new/data.yaml
                        # map to same id if within merged range
                        new_cid = cid if cid < len(merged_names) else None
                    if new_cid is None:
                        print(f'Warning: cannot map class id {cid} in {src_lbl}')
                        continue
                    remapped.append(' '.join([str(new_cid)] + parts[1:]))
            else:
                remapped = None

            planned_ops.append((src_img, dst_img, src_lbl if os.path.exists(src_lbl) else None, dst_lbl if remapped is not None else None))
            summary[subset + '_images'] += 1
            if remapped is not None:
                summary[subset + '_labels'] += 1

    # report planned operations
    print('\nPlanned operations:')
    for src_img, dst_img, src_lbl, dst_lbl in planned_ops:
        print(f'COPY: {os.path.relpath(src_img, root)} -> {os.path.relpath(dst_img, root)}')
        if src_lbl and dst_lbl:
            print(f'  remap label: {os.path.relpath(src_lbl, root)} -> {os.path.relpath(dst_lbl, root)}')
        elif src_lbl and not dst_lbl:
            print(f'  label present but could not remap: {os.path.relpath(src_lbl, root)}')
        else:
            print('  no label file')

    print('\nSummary:')
    for k, v in summary.items():
        print(f'  {k}: {v}')

    print('\nMerged class count will be:', len(merged_names))

    if dry_run:
        print('\nDry-run mode: no files were copied. Re-run with --apply to perform the merge and update data.yaml.')
        return 0

    # APPLY changes
    # 1) backup data.yaml
    bak = main_data_yaml + '.bak'
    print('\nApplying changes...')
    shutil.copy2(main_data_yaml, bak)
    print('  backed up', main_data_yaml, '->', bak)

    # 2) perform copies and write remapped labels
    for src_img, dst_img, src_lbl, dst_lbl in planned_ops:
        shutil.copy2(src_img, dst_img)
        if src_lbl and dst_lbl:
            # remap label file again (read, remap -> dst_lbl)
            with open(src_lbl, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            remapped = []
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                new_cid = new_to_merged.get(cid, cid)
                remapped.append(' '.join([str(new_cid)] + parts[1:]))
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.write('\n'.join(remapped) + ('\n' if remapped else ''))

    # 3) update main data.yaml names and nc
    new_main_cfg = dict(main_cfg)
    new_main_cfg['nc'] = len(merged_names)
    new_main_cfg['names'] = merged_names
    # write YAML
    if yaml:
        with open(main_data_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(new_main_cfg, f, sort_keys=False, allow_unicode=True)
    else:
        # simple writer
        with open(main_data_yaml, 'w', encoding='utf-8') as f:
            # preserve train/val/test from original if present
            for key in ('train', 'val', 'test'):
                if key in main_cfg:
                    f.write(f"{key}: {main_cfg[key]}\n")
            f.write('\n')
            f.write(f'nc: {len(merged_names)}\n')
            f.write('names: [' + ', '.join([f"'{n}'" for n in merged_names]) + ']\n')

    print('Merge completed. Updated', main_data_yaml)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Perform copy and update data.yaml')
    args = parser.parse_args()
    rc = main(dry_run=not args.apply)
    sys.exit(rc)
