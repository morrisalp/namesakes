"""
Download Wikipedia infobox images for celebrities listed in data/wp_celeb_fame.csv.

Output: data_prep/faces/{name_slug}/infobox.jpg  (max 256px on longest side)
        data_prep/faces/{name_slug}/license.json  (license metadata)
"""

import csv
import io
import json
import logging
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm.auto import tqdm

API_URL     = "https://en.wikipedia.org/w/api.php"
HEADERS     = {"User-Agent": "NamesakesCelebDataset/1.0"}
MAX_SIZE    = 256
SLEEP_S     = 3.0
MAX_RETRIES = 5

INPUT      = Path(__file__).parent.parent.parent / "data" / "wp_celeb_fame.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "faces"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def get_with_backoff(url: str, **kwargs) -> requests.Response:
    """GET with exponential backoff on 429 responses."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        r = requests.get(url, **kwargs)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        wait = delay * (2 ** attempt)
        log.warning("429 on %s — retrying in %.0fs", url, wait)
        time.sleep(wait)
    r.raise_for_status()
    return r  # unreachable, for type checker


def get_image_info(title: str) -> tuple[str, str] | None:
    """Return (thumbnail_url, image_name) for the page's infobox image, or None."""
    r = get_with_backoff(
        API_URL,
        headers=HEADERS,
        params={
            "action": "query",
            "prop": "pageimages",
            "piprop": "thumbnail|name",
            "pithumbsize": MAX_SIZE,
            "titles": title,
            "format": "json",
        },
        timeout=20,
    )
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    thumbnail = page.get("thumbnail")
    image_name = page.get("pageimage")  # canonical name without "File:" prefix
    if thumbnail and image_name:
        return thumbnail["source"], image_name
    return None


def get_license(image_name: str) -> dict:
    """Return license metadata for a File: page."""
    r = get_with_backoff(
        API_URL,
        headers=HEADERS,
        params={
            "action": "query",
            "titles": f"File:{image_name}",
            "prop": "imageinfo",
            "iiprop": "extmetadata",
            "format": "json",
        },
        timeout=20,
    )
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    imageinfo = page.get("imageinfo") or [{}]
    meta = imageinfo[0].get("extmetadata", {})
    return {
        "image_name": f"File:{image_name}",
        "license":     meta.get("LicenseShortName", {}).get("value", ""),
        "license_url": meta.get("LicenseUrl",       {}).get("value", ""),
        "artist":      meta.get("Artist",            {}).get("value", ""),
    }


def download_and_save(title: str) -> bool:
    """Fetch image + license and save both. Returns True on success."""
    slug         = title.replace(" ", "_")
    out_path     = OUTPUT_DIR / slug / "infobox.jpg"
    license_path = OUTPUT_DIR / slug / "license.json"
    if out_path.exists() and license_path.exists():
        return True  # already done; skip without sleeping

    time.sleep(SLEEP_S)

    try:
        info = get_image_info(title)
    except Exception as e:
        log.warning("Failed to get image info for %r: %s", title, e)
        return False

    if info is None:
        log.warning("No infobox image for %r", title)
        return False

    url, image_name = info
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        try:
            r = get_with_backoff(url, headers=HEADERS, timeout=30)
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            log.warning("Failed to download/open image for %r: %s", title, e)
            return False

        w, h = img.size
        if max(w, h) > MAX_SIZE:
            scale = MAX_SIZE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.save(out_path, "JPEG", quality=90)

    if not license_path.exists():
        try:
            license_data = get_license(image_name)
        except Exception as e:
            log.warning("Failed to get license for %r: %s", title, e)
            return False
        with license_path.open("w", encoding="utf-8") as f:
            json.dump(license_data, f, indent=2)

    return True


def main():
    with INPUT.open(newline="", encoding="utf-8") as f:
        names = [row["name"] for row in csv.DictReader(f)]

    print(f"Downloading images for {len(names)} celebrities → {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    for name in tqdm(names, unit="person"):
        try:
            ok = download_and_save(name)
        except Exception as e:
            log.warning("Unhandled error for %r: %s", name, e)
            ok = False
        if ok:
            success += 1

    print(f"Done. Saved {success}/{len(names)} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
