"""
Download ACN-Data
Fetches the Caltech EV charging dataset from Dec 2020 to Jan 2021 using the acndata API.
Saves the results into data/raw/sessions.csv for the preprocessing pipeline.
"""

import os
import json
import requests
import urllib.parse
import pandas as pd

API_TOKEN = "DEMO_TOKEN"
BASE_URL = "https://ev.caltech.edu/api/v1"
ENDPOINT = "/sessions/caltech"
# Date range according to the replication specs
START_STR = "Tue, 01 Dec 2020 00:00:00 GMT"
END_STR   = "Mon, 01 Feb 2021 00:00:00 GMT"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
OUT_CSV  = os.path.join(RAW_DIR, "sessions.csv")

def download_data():
    where_clause = f'{{"connectionTime": {{"$gte": "{START_STR}", "$lte": "{END_STR}"}}}}'
    query = f"?where={urllib.parse.quote(where_clause)}"
    
    url = BASE_URL + ENDPOINT + query
    print(f"Fetching data from {START_STR} to {END_STR}...")
    
    sessions = []
    
    while url:
        print(f" Requesting: {url.split('?')[-1][:50]}...")
        res = requests.get(url, auth=(API_TOKEN, ""))
        if res.status_code != 200:
            print(f"[ERROR] API returned {res.status_code}: {res.text}")
            break
            
        data = res.json()
        items = data.get("_items", [])
        sessions.extend(items)
        
        # Pagination
        links = data.get("_links", {})
        next_page = links.get("next", {}).get("href")
        if next_page:
            url = BASE_URL + "/" + next_page
        else:
            url = None

    print(f"Downloaded {len(sessions)} sessions.")
    if not sessions:
        return

    df = pd.DataFrame(sessions)
    if "userInputs" in df.columns:
        df["userInputs"] = df["userInputs"].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    if "_id" in df.columns:
        df["sessionID"] = df["_id"]

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")

if __name__ == "__main__":
    download_data()
