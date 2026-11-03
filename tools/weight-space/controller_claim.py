"""Claim a fresh controller output directory before writing mutable status."""
import datetime
import json
import os
from pathlib import Path
import re
import sys


def claim(directory, protocol_sha256):
    root=Path(directory).resolve(strict=True)
    if not root.is_dir() or not re.fullmatch(r'[0-9a-f]{64}',protocol_sha256):
        raise ValueError('existing output directory and protocol SHA-256 required')
    owner=dict(pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               protocol_sha256=protocol_sha256,argv=sys.argv)
    # O_EXCL is atomic across processes. Keep the receipt after completion;
    # reruns need a fresh directory, including after an interrupted controller.
    fd=os.open(root/'controller-owner.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as out:
        json.dump(owner,out,indent=2);out.write('\n')
    return owner
