"""Download only the official right-hand model and meshes, at a pinned commit."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "0d19cac602f46456b819e4b6a2c09a74982c9a3e"
BASE = f"https://raw.githubusercontent.com/sharpa-robotics/sharpa-urdf-usd-xml/{COMMIT}/"
DEST = ROOT / "assets" / "sharpa"
HAND = "wave_01/right_sharpa_wave/"


def fetch(path):
    dest = DEST / path
    if not dest.exists():
        data = urllib.request.urlopen(BASE + path, timeout=60).read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return {"path": path, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "bytes": dest.stat().st_size}


if __name__ == "__main__":
    records = [fetch(HAND + "right_sharpa_wave.xml")]
    tree = ET.parse(DEST / HAND / "right_sharpa_wave.xml")
    files = [HAND + "meshes/" + e.attrib["file"] for e in tree.findall("./asset/mesh")]
    files += ["README.md", "LICENSE.txt", "NOTICE.txt", HAND + "right_sharpa_wave.urdf", HAND + "right_sharpa_wave.usda", HAND + "meshes/right_hand_C_MC_visual.STL"]
    # Preserve the USD dependency tree for a later Isaac Lab port.
    files += [HAND + "right_sharpa_wave/right_sharpa_wave.usd"]
    files += [HAND + "right_sharpa_wave/configuration/right_sharpa_wave_" + name + ".usd" for name in ("base", "physics", "robot", "sensor")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records += list(pool.map(fetch, sorted(set(files))))
    manifest = {"repository": "https://github.com/sharpa-robotics/sharpa-urdf-usd-xml", "commit": COMMIT, "files": records}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Fetched {len(records)} files ({sum(r['bytes'] for r in records) / 1e6:.1f} MB), commit {COMMIT}")
