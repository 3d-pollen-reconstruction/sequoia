#!/usr/bin/env python3
import os
import glob
import time
import shutil
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import argparse
import threading
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/131.0.0.0 Safari/537.36")

CSV_FILE = "3d_pollen_library.csv"
RAW_DIR = os.path.join(os.getcwd(), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

downloaded_meshes = {}
downloaded_meshes_lock = threading.Lock()

def sanitize_title(title):
    """Sanitize a title string to be filesystem‐friendly."""
    title = title.strip()
    title = re.sub(r"\s+", "_", title)
    title = re.sub(r"[^\w\-]", "", title)
    if not title:
        title = "unknown"
    return title

def extract_fields(hit):
    """
    Extract fields from a single hit returned by the API.
    Ensures that the "id" field is a single value.
    """
    fields = hit["fields"]
    if len(fields["id"]) != 1:
        raise ValueError("Expected exactly one id field")
    fields_dict = {"id": fields["id"][0]}
    for key, value in fields.items():
        if key != "id":
            fields_dict[key] = value[0] if isinstance(value, list) and len(value) == 1 else value
    return fields_dict

def fetch_records():
    """
    Fetch all records from the NIH 3D Print Exchange API using pagination.
    Displays a tqdm progress bar and returns a list of record dictionaries.
    """
    all_records = []
    start = 0
    step = 48

    init_url = (
        f"https://3d.nih.gov/api/search/type:entry%20AND%20submissionstatus:%22Published%22"
        f"%20AND%20collectionid:33?start={start}&size={step}&sort=created%20desc"
    )
    response = requests.get(init_url, headers={"User-Agent": USER_AGENT})
    if response.status_code != 200:
        print(f"Error: Failed to fetch data from API (status code: {response.status_code}).")
        return all_records

    obj = response.json()
    total_found = obj["hits"]["found"]

    pbar = tqdm(total=total_found, desc="Fetching records", unit="record")
    while start < total_found:
        url = (
            f"https://3d.nih.gov/api/search/type:entry%20AND%20submissionstatus:%22Published%22"
            f"%20AND%20collectionid:33?start={start}&size={step}&sort=created%20desc"
        )
        response = requests.get(url, headers={"User-Agent": USER_AGENT})
        if response.status_code != 200:
            print(f"Error: Failed to fetch data from {url} (status code: {response.status_code}).")
            break

        obj = response.json()
        hits = obj["hits"]["hit"]
        for hit in hits:
            try:
                record = extract_fields(hit)
                all_records.append(record)
            except Exception as e:
                print(f"Error extracting fields: {e}")
        start += step
        pbar.update(step)
    pbar.close()
    return all_records

def wait_for_download(download_path, timeout):
    """
    Wait for an STL file to appear in download_path.
    Also wait until any temporary .crdownload files are finished.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        stl_files = glob.glob(os.path.join(download_path, "*.stl"))
        crdownload_files = glob.glob(os.path.join(download_path, "*.crdownload"))
        if stl_files and not crdownload_files:
            return stl_files[0]
        time.sleep(0.5)
    return None

def download_model(row, driver, download_path, timeout=30):
    """
    Downloads the mesh corresponding to the CSV row.
    Uses CSV fields for naming: destination filename will be
       {row.id}_{sanitized_title}.stl
    If a duplicate is detected (same sanitized title), copies the
    already downloaded file to the new destination.
    Returns None on success, or the row (for retry) on failure.
    """
    row_dict = row._asdict()
    try:
        # Build the canonical name from CSV.
        row_id = row_dict["id"]
        # Use the CSV "title" field if available; otherwise, use a default.
        csv_title = row_dict.get("title", "unknown")
        canonical = sanitize_title(csv_title)
        dest_filename = f"{row_id}_{canonical}.stl"
        destination = os.path.join(RAW_DIR, dest_filename)

        url = f"https://3d.nih.gov/entries/{row_id}"
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Click the 'Download' link.
        download_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[text()='Download']"))
        )
        driver.execute_script("arguments[0].click();", download_link)

        # Select the STL option.
        stl_label = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//label[text()='stl']"))
        )
        stl_label.click()

        # Click the button to download files.
        download_files_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "downloadfilesBtn"))
        )
        download_files_btn.click()

        # Agree to the terms.
        terms_checkbox = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "termsCheckbox"))
        )
        terms_checkbox.click()

        # Click the final Download button.
        final_download_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Download']"))
        )
        final_download_btn.click()

        # Wait for the file to be fully downloaded.
        downloaded_file = wait_for_download(download_path, timeout)
        if not downloaded_file:
            raise TimeoutError("Download timed out - no STL file found")

        # With duplicates: use the canonical title as key.
        with downloaded_meshes_lock:
            if canonical in downloaded_meshes:
                # Duplicate: copy the already downloaded file.
                src = downloaded_meshes[canonical]
                shutil.copy(src, destination)
                print(f"[{row_id}] Duplicate mesh detected; copied from {os.path.basename(src)} to {dest_filename}")
                # Remove the temporary download.
                os.remove(downloaded_file)
            else:
                # First time: move the downloaded file and record it.
                shutil.move(downloaded_file, destination)
                downloaded_meshes[canonical] = destination
                print(f"[{row_id}] Downloaded and saved as {dest_filename}")

        time.sleep(2)
        return None
    except Exception as e:
        print(f"[{row_dict['id']}] Error downloading mesh: {e}")
        return row

def create_driver(row_dict, headless=True):
    """
    Creates and configures a Selenium Chrome driver with a unique temporary download directory.
    Returns the driver and its associated download path.
    """
    download_path = os.path.abspath(os.path.join(os.getcwd(), f"temp_{row_dict['id']}"))
    os.makedirs(download_path, exist_ok=True)
    print(f"Driver {row_dict['id']} download path: {download_path}")

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": download_path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False,
            "plugins.always_open_pdf_externally": True,
        },
    )
    
    if headless:
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-certificate-errors-spki-list')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--enable-features=NetworkService,NetworkServiceInProcess")
    chrome_options.add_argument(f"user-agent={USER_AGENT}")

    driver = webdriver.Chrome(options=chrome_options)
    driver.maximize_window()
    
    if headless:
        try:
            driver.execute_cdp_cmd('Page.setDownloadBehavior', {
                'behavior': 'allow',
                'downloadPath': download_path
            })
            print(f"Headless mode: download behavior set to {download_path}")
        except Exception as e:
            print(f"ERROR setting headless download behavior: {e}")
    
    return driver, download_path

def process_partition(driver, download_path, rows, timeout, pbar):
    """
    Process a list (partition) of CSV rows sequentially using a single driver.
    For each row, update the shared progress bar.
    Returns a list of rows for which the download failed.
    """
    failed = []
    for row in rows:
        result = download_model(row, driver, download_path, timeout)
        if result is not None:
            failed.append(result)
        pbar.update(1)
    return failed

def main():
    parser = argparse.ArgumentParser(description="Download 3D pollen STL models.")
    parser.add_argument("--no-headless", action="store_false", dest="headless",
                        help="Run with browser window visible (not headless).", default=True)
    args = parser.parse_args()
    headless = args.headless

    # Load (or create) the CSV.
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        print(f"Loaded CSV with {df.shape[0]} records.")
    else:
        print("CSV file not found; fetching records from API...")
        records = fetch_records()
        if not records:
            print("No records fetched. Exiting.")
            return
        df = pd.DataFrame(records)
        df.to_csv(CSV_FILE, index=False)
        print(f"Fetched and saved {df.shape[0]} records.")

    rows = list(df.itertuples())
    total_records = len(rows)
    max_workers = 8

    # Create max_workers drivers and assign each a fixed subset of rows.
    drivers = []
    for i in range(max_workers):
        drv, dl_path = create_driver({"id": f"worker_{i}"}, headless=headless)
        drivers.append((drv, dl_path))
    partitions = [rows[i::max_workers] for i in range(max_workers)]

    pbar = tqdm(total=total_records, desc="Downloading models", unit="model")

    # Process partitions with retries.
    max_retries = 3
    retry_count = 0
    timeout = 60
    while retry_count <= max_retries and any(partitions):
        all_failed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(max_workers):
                if partitions[i]:
                    futures.append(executor.submit(process_partition,
                                                   drivers[i][0],
                                                   drivers[i][1],
                                                   partitions[i],
                                                   timeout,
                                                   pbar))
            for future in concurrent.futures.as_completed(futures):
                failed_rows = future.result()
                all_failed.extend(failed_rows)
        if all_failed:
            retry_count += 1
            timeout *= 2
            print(f"\nRetrying {len(all_failed)} failed downloads (attempt {retry_count + 1})...")
            partitions = [all_failed[i::max_workers] for i in range(max_workers)]
        else:
            break
    pbar.close()

    # Clean up drivers.
    for driver, download_path in drivers:
        driver.quit()
        try:
            shutil.rmtree(download_path)
        except Exception:
            pass
    
    # total files is len of RAW_DIR of files with .stl ending
    total_files = len([f for f in os.listdir(RAW_DIR) if f.endswith('.stl')])
    unique_meshes = len(downloaded_meshes)
    print("\nDownload Summary:")
    print(f"  Total CSV records: {total_records}")
    print(f"  Files in RAW directory: {total_files}")
    print(f"  Unique meshes (by title): {unique_meshes}")

if __name__ == '__main__':
    main()
