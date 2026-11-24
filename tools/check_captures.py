#!/usr/bin/env python3
"""Validate a completed benchmark's matched capture records without third-party packages.

PNG is a display-only view; HDR PFM is required for quantitative image error.
"""
import argparse
import csv
import struct
import zlib
from collections import defaultdict
from pathlib import Path

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def image_size(path):
    with path.open('rb') as f:
        prefix = f.read(32)
        if prefix.startswith(PNG_SIGNATURE):
            if prefix[12:16] != b'IHDR' or len(prefix) < 24:
                raise ValueError(f'{path}: missing PNG IHDR')
            return struct.unpack('>II',prefix[16:24])
        if prefix.startswith(b'PF\n'):
            f.seek(0)
            assert f.readline().strip() == b'PF'
            w,h = map(int,f.readline().split())
            scale = float(f.readline())
            if scale >= 0: raise ValueError(f'{path}: expected little-endian PFM')
            # PFM pixel data are written exactly once, in RGB float32.
            header = f.tell()
            if path.stat().st_size - header != w*h*3*4:
                raise ValueError(f'{path}: wrong HDR byte count')
            return w,h
        raise ValueError(f'{path}: unknown image encoding')


def check(directory):
    directory = Path(directory)
    index = directory/'captures'/'index.csv'
    if not index.is_file(): raise ValueError(f'No capture index at {index}')
    with index.open(newline='',encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows: raise ValueError('Capture index is empty')
    methods = defaultdict(set)
    names = set()
    for row in rows:
        name = row['file']
        if name in names: raise ValueError(f'Duplicate capture: {name}')
        names.add(name)
        path = (directory/'captures'/name).resolve()
        if path.parent != (directory/'captures').resolve():
            raise ValueError(f'Unsafe filename: {name}')
        if not path.is_file(): raise ValueError(f'Missing: {path}')
        width, height = image_size(path)
        if (width,height) != (int(row['width']),int(row['height'])):
            raise ValueError(f'{name}: dimension mismatch')
        if path.suffix[1:] != row['format']:
            raise ValueError(f'{name}: format mismatch')
        if row['format']=='png':
            methods[(row['camera'],row['width'],row['height'],row['lights'])].add(row['method'])
    required={'radiance-budget','parallel-feature','full-rate-direct','full-rate-reference'}
    for case,present in methods.items():
        if present != required:
            raise ValueError(f'Unmatched view {case}: {sorted(present)}')
    if not methods: raise ValueError('No matched PNG comparisons')
    print(f'Validated {len(rows)} captured files in {len(methods)} matched camera/load view(s)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    check(args.results)
