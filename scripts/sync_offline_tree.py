"""Parallel, resumable offline tree upload over an existing SSH control connection."""
import argparse
import concurrent.futures
from pathlib import Path
import shlex
import subprocess


def sync(source, destination, host, port=22, control=None, workers=8):
    source = Path(source).resolve()
    options = ['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30',
               '-o', 'ServerAliveCountMax=6', '-p', str(port)]
    if control:
        options += ['-S', str(control)]
    subprocess.run(options + [host, 'mkdir -p ' + shlex.quote(destination)], check=True)
    buckets = [[] for _ in range(workers)]
    sizes = [0] * workers
    files = sorted([p for p in source.rglob('*') if p.is_file()], key=lambda p: p.stat().st_size, reverse=True)
    for path in files:
        i = sizes.index(min(sizes))
        buckets[i].append(path.relative_to(source).as_posix())
        sizes[i] += path.stat().st_size
    def send(bucket):
        if not bucket:
            return
        subprocess.run(['rsync', '-a', '--partial', '--append-verify', '--from0', '--files-from=-',
            '-e', shlex.join(options), str(source) + '/', host + ':' + destination.rstrip('/') + '/'],
            input=b'\0'.join(path.encode() for path in bucket) + b'\0', check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(send, buckets))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, default=22)
    p.add_argument('--control', type=Path)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    sync(args.source, args.destination, args.host, args.port, args.control, args.workers)
