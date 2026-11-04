"""Checksum-pinned downloads into disposable caches; never untrusted model uploads."""
import os
from pathlib import Path
import tempfile
import urllib.request
from urllib.parse import urlparse
from .core import digest


def fetch_asset(asset,cache):
    url=asset['url']; expected=asset['sha256']; budget=asset['max_bytes']
    if not isinstance(budget,int) or not 1<=budget<=32_000_000:
        raise ValueError('Asset byte budget must be 1..32000000')
    parsed=urlparse(url)
    release=parsed.hostname=='github.com' and parsed.path.startswith('/Ahmdullah16/Mini-Project-ML-Lab-RSSR/releases/download/v0.3.0-rgb-pilots/')
    if parsed.scheme!='https' or not (parsed.hostname=='raw.githubusercontent.com' or release):
        raise ValueError('Only pinned public GitHub artifacts and the versioned project pilot release are allowed')
    if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Invalid asset checksum')
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    target=cache/expected
    if target.exists() and digest(target)==expected: return target
    temporary=None
    try:
        with urllib.request.urlopen(url,timeout=30) as response, tempfile.NamedTemporaryFile(dir=cache,delete=False) as out:
            temporary=Path(out.name); count=0
            while chunk:=response.read(min(65536,budget-count+1)):
                count+=len(chunk)
                if count>budget: raise ValueError('Asset download exceeded byte budget')
                out.write(chunk)
        if digest(temporary)!=expected: raise ValueError('Downloaded asset checksum mismatch')
        os.replace(temporary,target)
        return target
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()
