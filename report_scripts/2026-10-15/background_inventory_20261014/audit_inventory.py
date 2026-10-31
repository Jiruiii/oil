from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import csv,json,re,hashlib,time
import numpy as np
import rasterio
REPO=Path(__file__).resolve().parents[3]; ROOT=REPO.parent  # outputs stay outside git
OUT=ROOT/'reports/2026-10-15/background_inventory_20261014'
SPLIT=REPO/'data_split/patch_level_gt_expand'; BGP=REPO/'data_split/patch_level_gt_expand_bg/bg_coords.tsv'
pat=re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')
splits={}; gt=set(); scenesets={}
for fold in (1,2,3):
 for split in ('train','val','test'):
  keys=[(m[1],int(m[2]),int(m[3])) for line in (SPLIT/f'{split}_fold{fold}.txt').read_text().splitlines() if (m:=pat.fullmatch(line.strip()))]
  assert len(keys)==len(set(keys))
  splits[fold,split]=keys;gt.update(keys);scenesets[fold,split]={k[0] for k in keys}
for fold in (1,2,3):
 for a,b in [('train','val'),('train','test'),('val','test')]: assert not scenesets[fold,a]&scenesets[fold,b]
with BGP.open() as f:bg={(r['scene'],int(r['x']),int(r['y'])) for r in csv.DictReader(f,delimiter='\t')}
assert not gt&bg
scenes=sorted({k[0] for k in gt}); byscene=defaultdict(list)
for key in sorted(gt|bg):byscene[key[0]].append(key)
frozen={}
with (BGP.parent/'stats.csv').open() as f:
 for r in csv.DictReader(f):
  m=re.search(r'(\d+) cands',r['msg']);frozen[r['scene']]=int(m[1]) if m else 0

def classify(a):
 n0=int(np.count_nonzero(a==0));n1=int(np.count_nonzero(a==1));n2=int(np.count_nonzero(a==2));n255=int(np.count_nonzero(a==255))
 return n0,n1,n2,n255

def scene_audit(scene):
 with rasterio.open(ROOT/'mask'/f'{scene}.tif') as ds:a=ds.read(1)
 h,w=a.shape;pc=classify(a);assert sum(pc)==a.size
 candidates=0;strict=0;matched=0;strict1=0
 for y in range(0,h-256+1,512):
  for x in range(0,w-256+1,512):
   t=a[y:y+256,x:x+256]
   if np.any(t==0):continue
   ignored=np.count_nonzero(t==255)
   if ignored>=32768:continue
   valid=np.count_nonzero((t==1)|(t==2))
   if valid<.3*65536:continue
   candidates+=1
   if ignored==0:strict+=1
   if np.all(t==1):strict1+=1
   if (scene,x,y) in bg:matched+=1
 patches=[]
 for key in byscene[scene]:
  _,x,y=key;t=a[y:y+256,x:x+256]
  n0,n1,n2,n255=classify(t);n255+=65536-t.size
  patches.append(dict(scene=scene,x=x,y=y,source='bg' if key in bg else 'gt',oil=n0,background=n1,others=n2,unlabeled=n255,has_oil=int(n0>0),passes_bg_rule=int(n0==0 and n1+n2>=.3*65536 and n255<32768),fully_annotated_negative=int(n0==0 and n255==0)))
 return dict(scene=scene,height=h,width=w,oil_pixels=pc[0],background_pixels=pc[1],others_pixels=pc[2],unlabeled_pixels=pc[3],grid_candidates=candidates,grid_fully_annotated_negatives=strict,grid_background1_only=strict1,frozen_grid_candidates=frozen[scene],baseline_bg=len([k for k in bg if k[0]==scene]),baseline_on_grid=matched),patches
start=time.time();scenerows=[];patchrows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 for i,(sr,pr) in enumerate(pool.map(scene_audit,scenes),1):
  scenerows.append(sr);patchrows.extend(pr)
  if i%50==0 or i==len(scenes):print(f'Audited {i}/{len(scenes)} scenes ({time.time()-start:.1f}s)',flush=True)
idx={(r['scene'],r['x'],r['y']):r for r in patchrows};si={r['scene']:r for r in scenerows}
foldrows=[]
for fold in (1,2,3):
 for split in ('train','val','test'):
  keys=splits[fold,split];ss=scenesets[fold,split];gp=[idx[k] for k in keys];bp=[idx[k] for k in bg if k[0] in ss]
  neg=sum(not p['has_oil'] for p in gp);bn=len(bp)
  foldrows.append(dict(fold=fold,split=split,scenes=len(ss),gt_total=len(gp),gt_oil=len(gp)-neg,gt_negative=neg,bg_list=bn,baseline_negative=neg+bn,baseline_total=len(gp)+bn,old_hnm_selected_bg=round(bn*.5) if split=='train' else bn,old_hnm_total_negative=neg+round(bn*.5) if split=='train' else neg+bn,mask_grid_candidates=sum(si[k]['grid_candidates'] for k in ss),grid_fully_annotated_negatives=sum(si[k]['grid_fully_annotated_negatives'] for k in ss),grid_background1_only=sum(si[k]['grid_background1_only'] for k in ss),gt_negative_passing_bg_rule=sum(p['passes_bg_rule'] for p in gp),bg_list_with_unlabeled=sum(p['unlabeled']>0 for p in bp)))
def counts(rows):return dict(total=len(rows),oil=sum(r['has_oil'] for r in rows),negative=sum(not r['has_oil'] for r in rows),fully_annotated_negative=sum(r['fully_annotated_negative'] for r in rows),passing_bg_rule=sum(r['passes_bg_rule'] for r in rows),with_unlabeled=sum(r['unlabeled']>0 for r in rows))
summary=dict(scene_count=len(scenes),gt=counts([r for r in patchrows if r['source']=='gt']),bg=counts([r for r in patchrows if r['source']=='bg']),combined=counts(patchrows),full_scene_pixels={k:sum(r[k] for r in scenerows) for k in ('oil_pixels','background_pixels','others_pixels','unlabeled_pixels')},grid_candidates=sum(r['grid_candidates'] for r in scenerows),grid_fully_annotated_negatives=sum(r['grid_fully_annotated_negatives'] for r in scenerows),grid_background1_only=sum(r['grid_background1_only'] for r in scenerows),frozen_grid_mismatches=[r['scene'] for r in scenerows if r['grid_candidates']!=r['frozen_grid_candidates']],bg_on_grid=sum(r['baseline_on_grid'] for r in scenerows),folds=foldrows,notes=['Counts are scene/x/y coordinate counts, not unique physical areas.','Grid: patch 256, stride 512; mask rule oil=0 pixels, labels1+2>=30%, label255<50%.','Grid candidates are mask-only; full image validity and annotation review not performed for expanded pool.','Baseline counts are dataset membership, not exact per-epoch exposure (drop_last=True, batch16).','HNM counts are provisional keep_fraction=0.5 settings, not completed training.'],sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BGP,*sorted(SPLIT.glob('*fold*.txt'))]})
for name,rows in [('by_scene.csv',scenerows),('existing_patch_labels.csv',patchrows),('by_fold.csv',foldrows)]:
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
