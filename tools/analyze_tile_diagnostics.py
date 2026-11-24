#!/usr/bin/env python3
"""Check recorded 8x8 feature-gate tiles and summarize actual error by shading rate."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

def analyze(path, budget):
    with path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError('No tile diagnostic rows')
    data=defaultdict(lambda:dict(tiles=0,pixels=0,violations=0,err_sq=0.,ref_sq=0.,max_abs=0.))
    coords=set()
    for r in rows:
        coord=(int(r['tile_x']),int(r['tile_y']))
        if coord in coords:raise ValueError(f'Duplicate tile {coord}')
        coords.add(coord)
        rate=int(round(float(r['rate'])))
        if rate not in (1,2,4):raise ValueError(f'Invalid rate {rate} at {coord}')
        d=data[rate]
        n=int(r['valid_pixels']);v=int(r['pixel_budget_violations']);mx=float(r['max_abs_luminance_error'])
        if not (0<n<=64 and 0<=v<=n and math.isfinite(mx) and mx>=0):raise ValueError('Invalid sample')
        d['tiles']+=1;d['pixels']+=n;d['violations']+=v
        d['max_abs']=max(d['max_abs'],mx)
        d['err_sq']+=float(r['rgb_squared_error']);d['ref_sq']+=float(r['rgb_squared_reference'])
    print(f'Tiles: {len(rows)} / pixels: {sum(d["pixels"] for d in data.values())} / absolute luminance budget: {budget}')
    for rate in sorted(data):
        d=data[rate]
        print(f'rate {rate}x{rate}: {d["tiles"]} tiles, '
              f'violation={d["violations"]/d["pixels"]:.5%}, '
              f'relative_RGB_RMSE={math.sqrt(d["err_sq"]/max(d["ref_sq"],1e-20)):.6f}, '
              f'max_abs_Y={d["max_abs"]:.6f}')
    print('Note: rates and violations are measured, not theoretical error guarantees.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tiles',type=Path)
    p.add_argument('--budget',type=float,default=0.025)
    args=p.parse_args();analyze(args.tiles,args.budget)
