"""
Script to store TEMPO V04 L3 tiles for a provided date range

Output: TEMPO level3 images stored in TEMPO_DIR
"""

import earthaccess
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import * 

def main():
    """request tiles to download from API"""
    try:
        earthaccess.login(strategy="environment")
        results = earthaccess.search_data(
          short_name="TEMPO_NO2_L3",
          version="V03",
          temporal=(TEMPO_START_DATE, TEMPO_END_DATE),
        )
        if not results:
            print("Download request returned no results")
            return
        os.makedirs(TEMPO_DIR, exist_ok=True)
        files = earthaccess.download(results, TEMPO_DIR)

    except Exception as e:
        print(f"ERROR: {e}")
    
if __name__ == "__main__":
    main()