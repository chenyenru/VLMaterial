# pip install requests

import os
import json
import base64
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import requests


def fetch_materials(
    query: str,
    *,
    api_url: str = "http://localhost:8000/search",
    top_k: int = 1,
    cache_dir: Union[str, Path] = ".material_cache",
    timeout: int = 60,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    Query the Flask CLIP search API and cache the returned image + script locally.

    Returns a list of dicts (length == top_k) with:
      - score: float
      - remote_image_path: str | None
      - remote_script_path: str | None
      - image_path_local: str | None  # cached file path (if image provided)
      - script_path_local: str | None # cached file path (if script provided)
      - meta_path_local: str          # cached metadata json

    Caching strategy:
      - Each result entry is keyed by a stable hash derived from
        (remote_script_path or remote_image_path or query+rank).
      - If files already exist in cache, they are reused.

    Server expectations:
      - /search accepts JSON with {text, top_k, return_image, return_script}
      - Response for top_k==1:
          {"path": ".../transpiled_render.jpg", "score": 0.81,
           "image_base64": "data:image/jpeg;base64,...",
           "script_path": ".../blender_full.py", "script_text": "..." }
        For top_k>1:
          {"results": [{...}, {...}], ...}
    """
    cache_dir = Path(cache_dir)
    (cache_dir / "images").mkdir(parents=True, exist_ok=True)
    (cache_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (cache_dir / "entries").mkdir(parents=True, exist_ok=True)

    s = session or requests.Session()

    # 1) Call API (ask for image + script to be embedded)
    payload = {
        "text": query,
        "top_k": int(top_k),
        "return_indices": False,
        "return_image": True,
        "return_script": True,
    }
    resp = s.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # 2) Normalize to a list of result dicts
    if "results" in data:   # top_k > 1
        items = data["results"]
    else:                   # top_k == 1
        items = [data]

    out: List[Dict[str, Any]] = []

    def _stable_key(item: Dict[str, Any], rank: int) -> str:
        # Prefer script path (directory identity), then image path, else query+rank
        key_src = (
            item.get("script_path")
            or item.get("path")                # image path from server
            or f"{query}::rank={rank}"
        )
        return hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]

    def _save_image_from_data_url(data_url: str, dest_no_ext: Path) -> Optional[Path]:
        """
        Accepts a data URL like: data:image/jpeg;base64,<payload>
        Saves to dest_no_ext with appropriate extension inferred from MIME.
        """
        try:
            if not data_url or not data_url.startswith("data:"):
                return None
            header, b64 = data_url.split(",", 1)
            # header example: data:image/jpeg;base64
            mime = header.split(";")[0].split(":")[1]  # image/jpeg
            ext = {
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
                "image/bmp": ".bmp",
                "image/tiff": ".tiff",
            }.get(mime, ".bin")
            raw = base64.b64decode(b64)
            dest = dest_no_ext.with_suffix(ext)
            if not dest.exists():
                dest.write_bytes(raw)
            return dest
        except Exception:
            return None

    def _save_text(text: Optional[str], dest: Path) -> Optional[Path]:
        if text is None:
            return None
        if not dest.exists():
            dest.write_text(text, encoding="utf-8")
        return dest

    # 3) Persist each result
    for rank, item in enumerate(items):
        key = _stable_key(item, rank)
        meta_path = cache_dir / "entries" / f"{key}.json"
        img_dest_no_ext = cache_dir / "images" / key           # extension decided by MIME
        script_dest = cache_dir / "scripts" / f"{key}.py"

        # Load from cache if meta exists and files are present
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                # honor existing cache if files are there
                img_local = Path(meta.get("image_path_local")) if meta.get("image_path_local") else None
                script_local = Path(meta.get("script_path_local")) if meta.get("script_path_local") else None
                if (img_local is None or img_local.exists()) and (script_local is None or script_local.exists()):
                    out.append(meta)
                    continue
            except Exception:
                pass  # fall through to rebuild

        # Otherwise, build fresh from API response
        img_local_path = None
        script_local_path = None

        # Save image if provided
        img_data_url = item.get("image_base64")
        if img_data_url:
            img_local = _save_image_from_data_url(img_data_url, img_dest_no_ext)
            img_local_path = str(img_local) if img_local else None

        # Save script if provided
        script_text = item.get("script_text")
        if script_text is not None:
            saved = _save_text(script_text, script_dest)
            script_local_path = str(saved) if saved else None

        # Prepare metadata record
        record = {
            "query": query,
            "score": float(item.get("score", 0.0)),
            "remote_image_path": item.get("path"),
            "remote_script_path": item.get("script_path"),
            "image_path_local": img_local_path,
            "script_path_local": script_local_path,
            "meta_path_local": str(meta_path),
        }

        # Write meta (idempotent)
        meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        out.append(record)

    return out

if __name__ == "__main__":
    import sys
    # Example usage
    # Get python arguments from command line
    if len(sys.argv) < 2:
        raise ValueError("Please provide a text query as argument")
    text_input = " ".join(sys.argv[1:])

    results = fetch_materials(
        text_input,
        api_url="http://brahmastra.ucsd.edu:3001/search",
        top_k=1,
        cache_dir=".material_cache",
    )

    for r in results:
        print(f"score={r['score']:.3f}")
        # print(" image (local): ", r["image_path_local"])
        # print(" script (local):", r["script_path_local"])
        # print("---")
        print(f"image_path_local={r['image_path_local']}")
        print(f"script_path_local={r['script_path_local']}")