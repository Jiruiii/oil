#!/usr/bin/env python
# [TODO/15] 下載 Prithvi-EO-2.0-300M 預訓 checkpoint（pin revision + SHA256 落 manifest）。
# 人工執行（訓練程式不隱式連網）：
#   /opt/conda/bin/python main/fetch_prithvi_checkpoint.py
# 已存在且 hash 與 manifest 相符時直接跳過；重新下載請先手動刪檔。
import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

REPO_ID = 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M'
REVISION = '9eb1b1102806593963daa333bcc491b1c6f8562f'   # 2026-07-13 pin（共識第 4 案）
FILENAME = 'Prithvi_EO_V2_300M.pt'
URL = f'https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{FILENAME}'
EXPECTED_SHA256 = 'faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7'
EXPECTED_SIZE_BYTES = 1326658396

DST_DIR = Path(__file__).resolve().parent.parent / 'pretrained_ckpt'
DST = DST_DIR / FILENAME
MANIFEST = DST_DIR / f'{FILENAME}.manifest.json'


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    if DST.exists():
        actual_size = DST.stat().st_size
        actual = sha256_of(DST)
        if actual_size != EXPECTED_SIZE_BYTES or actual != EXPECTED_SHA256:
            print(f'❌ 現有檔案不符官方 pin：size={actual_size}, sha256={actual}\n'
                  f'   expected size={EXPECTED_SIZE_BYTES}, sha256={EXPECTED_SHA256}\n'
                  f'   請手動處理後重跑', file=sys.stderr)
            return 1
        if MANIFEST.exists():
            recorded = json.loads(MANIFEST.read_text())
            if (recorded.get('sha256') == EXPECTED_SHA256 and
                    int(recorded.get('size_bytes', -1)) == EXPECTED_SIZE_BYTES and
                    recorded.get('revision') == REVISION):
                print(f'✅ 已存在且符合官方 pin，跳過下載：{DST}')
                return 0
        print(f'⚠ 檔案符合官方 pin，補寫/更新 manifest（不重新下載）')
    else:
        print(f'⏳ 下載 {URL}\n   → {DST}')
        tmp = DST.with_suffix('.pt.part')
        with urllib.request.urlopen(URL) as r, open(tmp, 'wb') as f:
            total = int(r.headers.get('Content-Length') or 0)
            done = 0
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
                done += len(b)
                if total:
                    print(f'\r   {done / 1e6:.0f}/{total / 1e6:.0f} MB', end='', flush=True)
        print()
        actual_size = tmp.stat().st_size
        digest = sha256_of(tmp)
        if actual_size != EXPECTED_SIZE_BYTES or digest != EXPECTED_SHA256:
            tmp.unlink(missing_ok=True)
            print(f'❌ 下載內容不符官方 pin：size={actual_size}, sha256={digest}', file=sys.stderr)
            return 1
        tmp.rename(DST)

    digest = sha256_of(DST)
    MANIFEST.write_text(json.dumps({
        'repo_id': REPO_ID,
        'revision': REVISION,
        'filename': FILENAME,
        'url': URL,
        'sha256': digest,
        'size_bytes': DST.stat().st_size,
        'expected_sha256': EXPECTED_SHA256,
        'expected_size_bytes': EXPECTED_SIZE_BYTES,
        'license': 'apache-2.0',
        'fetched_at': str(date.today()),
    }, indent=2) + '\n')
    print(f'✅ 完成：{DST}\n   sha256={digest}\n   manifest={MANIFEST}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
