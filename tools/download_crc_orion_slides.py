#!/usr/bin/env python3
"""Download the Orion CRC registered H&E WSIs from the public S3 bucket.

Convention follows the other colon cohorts (see slides/colon/README.md):
stream to `.part`, verify against the manifest (size + multipart md5 ETag),
rename only on success. Resumable: verified files are skipped, partial files
resume with an HTTP Range request.
"""
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://lin-2023-orion-crc.s3.amazonaws.com"
DEST = Path("/mnt/d/YC.Liu/slides/colon/crc_orion")
MANIFEST = DEST / "s3_he_manifest.json"
LOG = DEST / "download.log"
STATE = DEST / "download_state.jsonl"
WORKERS = 4

_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with _lock:
        print(line, flush=True)
        with LOG.open("a") as fh:
            fh.write(line + "\n")


def record(**kw) -> None:
    with _lock, STATE.open("a") as fh:
        fh.write(json.dumps(kw) + "\n")


def multipart_etag(path: Path, part_size: int, nparts: int) -> str:
    """Recompute S3's multipart ETag: md5 of the concatenated part md5s."""
    if nparts == 1:
        h = hashlib.md5()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()
    digests = []
    with path.open("rb") as fh:
        while True:
            part = fh.read(part_size)
            if not part:
                break
            digests.append(hashlib.md5(part).digest())
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{len(digests)}"


def verify(path: Path, rec: dict) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size != rec["size"]:
        return False, f"size {size} != {rec['size']}"
    got = multipart_etag(path, rec["part_size"] or rec["size"], rec["nparts"])
    if got != rec["etag"]:
        return False, f"etag {got} != {rec['etag']}"
    return True, "ok"


def fetch(rec: dict) -> dict:
    slide_id = rec["key"].split("/")[1]          # CRC01 ... CRC33_01 ... CRC40
    final = DEST / f"{slide_id}.ome.tif"
    part = DEST / f"{slide_id}.ome.tif.part"
    gb = rec["size"] / 1e9

    if final.exists():
        ok, why = verify(final, rec)
        if ok:
            log(f"SKIP  {slide_id}  already verified ({gb:.2f} GB)")
            record(slide_id=slide_id, status="skip_verified", **{"size": rec["size"]})
            return {"slide_id": slide_id, "status": "skip"}
        log(f"REDO  {slide_id}  existing file failed verify: {why}")
        final.unlink()

    url = f"{BUCKET}/{rec['key']}"
    for attempt in (1, 2, 3):
        t0 = time.time()
        log(f"GET   {slide_id}  ({gb:.2f} GB) attempt {attempt}")
        # -C - resumes a partial .part across attempts / reruns.
        cmd = ["curl", "-sS", "--fail", "-C", "-", "--retry", "5",
               "--retry-delay", "5", "--speed-limit", "10240",
               "--speed-time", "120", "-o", str(part), url]
        res = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if res.returncode != 0 and not (part.exists() and part.stat().st_size == rec["size"]):
            log(f"FAIL  {slide_id}  curl rc={res.returncode} {res.stderr.strip()[:200]}")
            continue
        ok, why = verify(part, rec)
        if ok:
            part.rename(final)
            mbps = rec["size"] / 1e6 / max(dt, 1e-9)
            log(f"OK    {slide_id}  {gb:.2f} GB in {dt/60:.1f} min ({mbps:.0f} MB/s) etag verified")
            record(slide_id=slide_id, status="ok", size=rec["size"], etag=rec["etag"],
                   seconds=round(dt, 1), source_key=rec["key"])
            return {"slide_id": slide_id, "status": "ok"}
        log(f"BAD   {slide_id}  verify failed: {why} — discarding and retrying")
        part.unlink(missing_ok=True)

    log(f"GIVEUP {slide_id} after 3 attempts")
    record(slide_id=slide_id, status="failed", source_key=rec["key"])
    return {"slide_id": slide_id, "status": "failed"}


def main() -> int:
    recs = json.loads(MANIFEST.read_text())
    total = sum(r["size"] for r in recs) / 1e9
    log(f"=== Orion CRC H&E download: {len(recs)} files, {total:.1f} GB -> {DEST} ===")
    # Largest first so the tail of the run is short files.
    recs.sort(key=lambda r: -r["size"])
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(fetch, recs))
    n_ok = sum(r["status"] in ("ok", "skip") for r in results)
    failed = [r["slide_id"] for r in results if r["status"] == "failed"]
    log(f"=== DONE: {n_ok}/{len(recs)} present and verified; failed: {failed or 'none'} ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
