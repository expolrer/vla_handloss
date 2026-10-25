"""Collect on the evaluation server; download only on the relay server.

An anchor-matched `none` episode can still require backgrounds, textures and clutter.
Do not silently replace missing assets with a clean scene.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import socket
import threading
import time
import urllib.parse


def asset_paths(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from asset_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from asset_paths(item)
    elif isinstance(value, str) and 'dex2bench_dataset/' in value:
        rel = value.split('dex2bench_dataset/', 1)[1]
        path = PurePosixPath(rel)
        if '..' in path.parts or path.is_absolute():
            raise ValueError('Unsafe asset path: ' + rel)
        yield path.as_posix()


def collect(anchors):
    import h5py
    paths, episodes = set(), []
    for task in sorted(Path(anchors).iterdir()):
        if not task.is_dir():
            continue
        for file in sorted(task.glob('*.hdf5')):
            with h5py.File(file) as source:
                raw = source['meta/scene_generalization_sample'][()]
                value = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            needed = sorted(set(asset_paths(value)))
            paths.update(needed)
            episodes.append(dict(task=task.name, anchor=file.name, assets=needed))
    return dict(episodes=episodes, paths=sorted(paths), collected_epoch=time.time(),
        note='Training-matched canonical anchors, not a held-out split')


def download(inventory, destination, state):
    import requests
    original = socket.getaddrinfo
    def ipv4(host, port, family=0, type=0, proto=0, flags=0):
        return original(host, port, socket.AF_INET if family == socket.AF_UNSPEC else family, type, proto, flags)
    socket.getaddrinfo = ipv4
    destination, state = Path(destination), Path(state)
    destination.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    thread = threading.local()
    def session():
        if not hasattr(thread, 'session'):
            thread.session = requests.Session()
        return thread.session
    def get_json(url, **kwargs):
        for attempt in range(6):
            try:
                response = session().get(url, timeout=(20, 90), **kwargs)
                response.raise_for_status()
                return response.json()
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(30, 2 ** attempt))
    prefixes, direct = set(), set()
    for rel in inventory['paths']:
        parts = PurePosixPath(rel).parts
        if parts[0] == 'Objects':
            prefixes.add('/'.join(parts[:2]))
        elif parts[:2] == ('scenes', 'ithor'):
            prefixes.add('/'.join(parts[:3]))
        else:
            direct.add(rel)
    rows = {}
    for prefix in sorted(prefixes):
        cache = state / (hashlib.sha256(prefix.encode()).hexdigest() + '.json')
        if cache.exists():
            files = json.loads(cache.read_text())
        else:
            files, page, seen = [], 1, set()
            while True:
                data = get_json('https://modelscope.cn/api/v1/datasets/Bench2Dex/Bench2Dex/repo/tree',
                    params=dict(Revision='master', Root=prefix, Recursive='true', PageNumber=page, PageSize=1000))['Data']
                batch = data.get('Files', [])
                new = [row for row in batch if row['Path'] not in seen]
                files.extend(new)
                seen.update(row['Path'] for row in new)
                if len(seen) >= data.get('TotalCount', len(batch)) or not batch:
                    break
                if not new:
                    raise ValueError('Asset API pagination did not advance: ' + prefix)
                page += 1
            cache.write_text(json.dumps(files))
        rows.update({row['Path']: row for row in files if row['Type'] != 'tree'})
    for rel in direct:
        rows.setdefault(rel, dict(Path=rel, Revision='master', Size=None, Sha256=''))
    (state / 'source_inventory.json').write_text(json.dumps(rows, indent=2))
    hashes = {}
    lock = threading.Lock()
    def fetch(item):
        rel, info = item
        target = destination / rel
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = info.get('Sha256')
        size = info.get('Size')
        cached = target.exists() and (size is None or target.stat().st_size == size)
        digest = hashlib.sha256(target.read_bytes()).hexdigest() if cached else None
        if not cached or (expected and digest != expected):
            for attempt in range(6):
                try:
                    revision = info.get('Revision') or 'master'
                    url = 'https://modelscope.cn/datasets/Bench2Dex/Bench2Dex/resolve/' + revision + '/' + urllib.parse.quote(rel, safe='/')
                    with session().get(url, stream=True, timeout=(20, 120)) as response:
                        response.raise_for_status()
                        temp = target.with_name(target.name + '.download')
                        h = hashlib.sha256()
                        with temp.open('wb') as f:
                            for block in response.iter_content(1024**2):
                                f.write(block)
                                h.update(block)
                        if size is not None and temp.stat().st_size != size:
                            raise ValueError('Asset size mismatch: ' + rel)
                        if expected and h.hexdigest() != expected:
                            raise ValueError('Asset SHA256 mismatch: ' + rel)
                        temp.replace(target)
                        digest = h.hexdigest()
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(min(30, 2 ** attempt))
        with lock:
            hashes[rel] = dict(bytes=target.stat().st_size, sha256=digest,
                source_revision=info.get('Revision'), official_sha256=expected or None)
            if len(hashes) % 20 == 0:
                print(json.dumps(dict(downloaded=len(hashes), total=len(rows))), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, rows.items()))
    manifest = dict(files=hashes, root_assets=inventory['paths'], episodes=inventory['episodes'],
        phase='ROOT_FOLDERS_DOWNLOADED', usd_transitive_check='PENDING', epoch=time.time())
    (state / 'anchor_assets_sha256.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--anchors', type=Path)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--asset-root', type=Path)
    p.add_argument('--state', type=Path)
    p.add_argument('--collect-only', action='store_true')
    args = p.parse_args()
    if args.collect_only:
        if not args.anchors:
            p.error('--anchors is required for collection')
        inventory = collect(args.anchors)
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(inventory, indent=2))
        print(json.dumps(dict(anchors=len(inventory['episodes']), unique_paths=len(inventory['paths']))))
    else:
        if not args.asset_root or not args.state:
            p.error('Download mode requires --asset-root and --state; run on the relay only')
        download(json.loads(args.inventory.read_text()), args.asset_root, args.state)


if __name__ == '__main__':
    main()
