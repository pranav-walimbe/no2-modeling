"""
Scraped tempo images for desired date range + version

Output: tempo images placed in TEMPO_DIR
"""

import os

import earthaccess

from config import (
    TEMPO_DIR,
    TEMPO_END_DATE,
    TEMPO_START_DATE,
    TEMPO_VERSION,
)
from prerequisites import require_earthdata_credentials


def main():
    require_earthdata_credentials()
    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name="TEMPO_NO2_L3",
        version=TEMPO_VERSION,
        temporal=(TEMPO_START_DATE, TEMPO_END_DATE),
    )
    if not results:
        print("Download request returned no results")
        return

    os.makedirs(TEMPO_DIR, exist_ok=True)
    existing = set(os.listdir(TEMPO_DIR))

    to_download = [r for r in results if os.path.basename(r.data_links()[0]) not in existing]
    print(f"{len(results) - len(to_download)} already present, downloading {len(to_download)}")

    if to_download:
        earthaccess.download(to_download, TEMPO_DIR)


if __name__ == "__main__":
    main()
