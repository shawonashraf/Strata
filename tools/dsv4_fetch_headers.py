#!/usr/bin/env python3
"""Fetch a (sharded) GGUF's headers over HTTP range requests and write them back as sparse local files.

    python tools/dsv4_fetch_headers.py REPO REV DIR OUTDIR

Only the header region of each shard is downloaded (metadata plus the tensor directory); the rest of every file
is a hole: the replica is ftruncate'd to the shard's real size, so its declared size and tensor ranges are the
original's without storing a byte of weight data.  `model_family_test --real OUT/<shard>` then runs the real
loader - architecture guard, geometry and the layout check across all shards - against the real file.

Written for the DeepSeek-V4-Flash stage 1 acceptance (docs/DEEPSEEK_V4_PLAN.md): the GA files are 155 GB across
five shards, and only their headers decide the layout tables.
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from quant_formats import RangeFile, Header  # noqa: E402


class Tee:
    """Wraps a file-like, keeping every byte it serves (the stream is read only forward)."""

    def __init__(self, fh):
        self.fh, self.buf = fh, bytearray()

    def read(self, n: int = -1) -> bytes:
        b = self.fh.read(n)
        self.buf += b
        return b

    def tell(self) -> int:
        return self.fh.tell()


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print("usage: dsv4_fetch_headers.py REPO REV DIR OUTDIR", file=sys.stderr)
        return 2
    repo, rev, d, outdir = argv[1], argv[2], argv[3], pathlib.Path(argv[4])
    tree = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/models/{repo}/tree/{rev}/{d}?recursive=false", timeout=60))
    files = sorted((e["path"], e["size"]) for e in tree if e["path"].endswith(".gguf"))
    if not files:
        print(f"{repo}:{rev}/{d}: no .gguf files", file=sys.stderr)
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    for path, size in files:
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
        tee = Tee(RangeFile(url))
        h = Header(url, tee)  # parses metadata + tensor directory; stops before any tensor data
        header_end, align = tee.tell(), h.alignment or 32
        data_start = (header_end + align - 1) // align * align
        if size < data_start:
            print(f"{path}: real size {size} is short of the header ({data_start} needed)", file=sys.stderr)
            return 1
        out = outdir / pathlib.Path(path).name
        with out.open("wb") as f:
            f.write(bytes(tee.buf))
            f.write(b"\0" * (data_start - header_end))
            f.truncate(size)  # sparse: the weight bytes stay a hole
        print(f"{out.name}: header {header_end} bytes, data at {data_start}, {len(h.tensors)} tensors,"
              f" {size} bytes on disk (sparse)")
    print(f"now run: model_family_test --real {sorted(outdir.glob('*.gguf'))[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
