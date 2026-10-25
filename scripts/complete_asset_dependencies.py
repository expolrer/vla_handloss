"""Read-only OpenUSD dependency traversal; download missing dependencies on the relay."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import socket
import time
import urllib.parse

BUILTIN_MDL = {'OmniPBR.mdl', 'OmniSurface.mdl', 'OmniGlass.mdl', 'OmniOpacity.mdl'}


def complete(asset_root, inventory, core_manifest=None):
    from pxr import Sdf, UsdUtils
    import requests
    root = Path(asset_root).resolve()
    paths = set(inventory['paths'])
    if core_manifest:
        paths.update(row['path'] for row in core_manifest['files'])
    original = socket.getaddrinfo
    def ipv4(host, port, family=0, type=0, proto=0, flags=0):
        return original(host, port, socket.AF_INET if family == socket.AF_UNSPEC else family, type, proto, flags)
    socket.getaddrinfo = ipv4
    visited, unresolved, builtin, pending = set(), [], set(), set(paths)
    def fetch(rel):
        path = root / rel
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://modelscope.cn/datasets/Bench2Dex/Bench2Dex/resolve/master/' + urllib.parse.quote(rel, safe='/')
        for attempt in range(6):
            try:
                with requests.get(url, timeout=(20, 90), stream=True) as response:
                    response.raise_for_status()
                    temp = path.with_name(path.name + '.download')
                    with temp.open('wb') as f:
                        for block in response.iter_content(1024**2):
                            f.write(block)
                    temp.replace(path)
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(30, 2 ** attempt))
    while pending:
        batch = sorted(pending - visited)
        pending.clear()
        if not batch:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, batch))
        for rel in batch:
            visited.add(rel)
            path = root / rel
            if path.suffix.lower() not in {'.usd', '.usda', '.usdc'}:
                continue
            layer = Sdf.Layer.FindOrOpen(str(path))
            if layer is None:
                raise ValueError('OpenUSD could not read: ' + rel)
            groups = UsdUtils.ExtractExternalReferences(str(path))
            for reference in (item for group in groups for item in group):
                if reference in BUILTIN_MDL:
                    builtin.add(reference)
                    continue
                resolved = Sdf.ComputeAssetPathRelativeToLayer(layer, reference)
                identifier, _ = Sdf.Layer.SplitIdentifier(resolved)
                candidate = Path(identifier).resolve()
                if not candidate.is_relative_to(root):
                    unresolved.append(dict(layer=rel, reference=reference, resolved=identifier))
                    continue
                dependency = candidate.relative_to(root).as_posix()
                if dependency not in visited:
                    pending.add(dependency)
        print(json.dumps(dict(visited=len(visited), pending=len(pending), unresolved=len(unresolved))), flush=True)
        if len(visited) > 100000:
            raise ValueError('Unexpectedly large USD dependency graph')
    files = {}
    for rel in sorted(visited):
        path = root / rel
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(block)
        files[rel] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
    return dict(files=files, usd_transitive_check='PASSED' if not unresolved else 'FAILED',
        unresolved=unresolved, runtime_builtin_mdl=sorted(builtin), epoch=time.time(),
        root_assets=inventory['paths'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--asset-root', type=Path, required=True)
    p.add_argument('--core-manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = complete(args.asset_root, json.loads(args.inventory.read_text()),
        json.loads(args.core_manifest.read_text()) if args.core_manifest else None)
    args.output.write_text(json.dumps(result, indent=2))
    if result['unresolved']:
        raise SystemExit('Unresolved external USD paths; no readiness receipt may be written')


if __name__ == '__main__':
    main()
