"""Refresh this evidence archive's file hashes after appending a result."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = root / "artifact-manifest.json"
files = {
    str(path.relative_to(root)): {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for path in sorted(root.rglob("*"))
    if path.is_file() and path != manifest
}
manifest.write_text(
    json.dumps(
        {"collected_at_utc": datetime.now(timezone.utc).isoformat(), "files": files},
        indent=2,
    )
    + "\n"
)
print(f"Hashed {len(files)} files ({sum(item['bytes'] for item in files.values())} bytes)")
