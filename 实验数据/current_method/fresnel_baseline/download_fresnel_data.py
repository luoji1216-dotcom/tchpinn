from __future__ import annotations

import argparse
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional


DATA_PAGE_URL = "https://iopscience.iop.org/article/10.1088/0266-5611/17/6/301/data"
ARTICLE_URL = "https://iopscience.iop.org/article/10.1088/0266-5611/17/6/301"
DEFAULT_FILE_NAME = "dielTM_dec8f.exp"


def _request_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def _download_binary(url: str, out_path: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        out_path.write_bytes(response.read())


def find_dataset_url(data_page_html: str, file_name: str) -> Optional[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', data_page_html, flags=re.IGNORECASE)
    for href in hrefs:
        decoded = urllib.parse.unquote(html.unescape(href))
        if file_name in decoded:
            return urllib.parse.urljoin(DATA_PAGE_URL, decoded)
    return None


def ensure_fresnel_data(
    output_dir: Path,
    file_name: str = DEFAULT_FILE_NAME,
    force: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file_name
    if out_path.exists() and not force:
        return out_path

    page_html = _request_text(DATA_PAGE_URL)
    (output_dir.parent / "iop_301_data_page.html").write_text(page_html, encoding="utf-8")

    data_url = find_dataset_url(page_html, file_name)
    if data_url is None:
        raise RuntimeError(
            f"Could not find {file_name!r} on the IOP data page. "
            f"Open {DATA_PAGE_URL} and download it manually if the page layout changed."
        )

    _download_binary(data_url, out_path)
    manifest = {
        "article_url": ARTICLE_URL,
        "data_page_url": DATA_PAGE_URL,
        "file_name": file_name,
        "download_url": data_url,
        "local_path": str(out_path),
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Institut Fresnel experimental dielectric-cylinder data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fresnel_2001",
        help="Directory where the .exp file will be stored.",
    )
    parser.add_argument(
        "--file-name",
        default=DEFAULT_FILE_NAME,
        help="Fresnel data file to download. Default: dielTM_dec8f.exp.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    args = parser.parse_args()

    out_path = ensure_fresnel_data(args.output_dir, args.file_name, force=args.force)
    print(f"Dataset ready: {out_path}")


if __name__ == "__main__":
    main()
