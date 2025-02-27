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

def download_model(row, driver, download_path, timeout=30):
    """
    Downloads a 3D model STL file from the NIH 3D Print Exchange website.
    Uses Selenium to automate the download process.
    Returns the row if an error occurs (so it can be retried), otherwise returns None.
    """
    row_dict = row._asdict()
    try:
        url = f"https://3d.nih.gov/entries/{row_dict['id']}"
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Click the 'Download' link
        download_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[text()='Download']"))
        )
        driver.execute_script("arguments[0].click();", download_link)

        # Select the STL option
        stl_label = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//label[text()='stl']"))
        )
        stl_label.click()

        # Click the button to download files
        download_files_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "downloadfilesBtn"))
        )
        download_files_btn.click()

        # Agree to the terms
        terms_checkbox = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "termsCheckbox"))
        )
        terms_checkbox.click()

        # Click the final Download button
        final_download_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Download']"))
        )
        final_download_btn.click()

        stl_file = None
        start_time = time.time()
        while time.time() - start_time < timeout:
            stl_files = glob.glob(os.path.join(download_path, "*.stl"))
            if stl_files:
                stl_file = stl_files[0]
                break
            time.sleep(0.5)

        if not stl_file:
            raise TimeoutError("Download timed out - no STL file found")

        # renaming the file with the model ID and moving it into raw
        original_name = os.path.basename(stl_file)
        new_name = f"{row_dict['id']}_{original_name}"
        destination = os.path.join(RAW_DIR, new_name)
        shutil.move(stl_file, destination)

        time.sleep(2)
    except Exception as e:
        return row

def create_driver(row_dict, headless=True):
    """
    Creates and configures a Selenium Chrome driver with a unique temporary download directory.
    Returns the driver and its associated download path.
    """
    download_path = os.path.abspath(os.path.join(os.getcwd(), f"temp_{row_dict['id']}"))
    os.makedirs(download_path, exist_ok=True)
    print(f"Download path for {row_dict['id']}: {download_path}")

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
            print(f"Successfully set download behavior for headless mode to: {download_path}")
        except Exception as e:
            print(f"ERROR setting download behavior in headless mode: {e}")
            print(f"This might prevent downloads from working properly.")
    
    return driver, download_path

def main():
    parser = argparse.ArgumentParser(description="Download 3D pollen STL models.")
    parser.add_argument("--no-headless", action="store_false", dest="headless",
                        help="Run with browser window visible (not headless).", default=True)
    args = parser.parse_args()
    headless = args.headless

    # Step 1: Create or load the CSV with metadata for Selenium downloads
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        print(f"Loaded existing CSV with {df.shape[0]} records.")
    else:
        print("CSV file not found; fetching records from API...")
        records = fetch_records()
        if not records:
            print("No records fetched. Exiting.")
            return
        df = pd.DataFrame(records)
        df.to_csv(CSV_FILE, index=False)
        print(f"Fetched {df.shape[0]} records and saved to CSV.")

    # Step 2: Set up Selenium drivers for parallel downloads
    max_workers = 8
    drivers = []
    for i in range(max_workers):
        drv, dl_path = create_driver({"id": f"worker_{i}"}, headless=headless)
        drivers.append((drv, dl_path))

    # Step 3: Download STL files with proper progress tracking
    rows_to_retry = list(df.itertuples())
    max_retries = 3
    retry_count = 0
    timeout = 60

    while retry_count <= max_retries and rows_to_retry:
        def download_with_progress(row, driver, download_path, timeout=30):
            result = download_model(row, driver, download_path, timeout)
            return result
            
        failed_rows = []
        total_downloads = len(rows_to_retry)
        
        with tqdm(total=total_downloads, desc=f"Downloading models (attempt {retry_count + 1})", unit="model") as pbar:
            future_to_row = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, row in enumerate(rows_to_retry):
                    worker_idx = i % max_workers
                    future = executor.submit(
                        download_with_progress,
                        row,
                        drivers[worker_idx][0],
                        drivers[worker_idx][1],
                        timeout,
                    )
                    future_to_row[future] = row
                
                for future in concurrent.futures.as_completed(future_to_row):
                    row = future_to_row[future]
                    try:
                        result = future.result()
                        if result is not None:
                            failed_rows.append(result)
                            print(f"Failed to download: {row.id}")
                    except Exception as exc:
                        print(f"Download generated an exception: {exc}")
                        failed_rows.append(row)
                    
                    pbar.update(1)
        
        rows_to_retry = failed_rows
        retry_count += 1
        timeout *= 2
        
        if failed_rows and retry_count <= max_retries:
            print(f"\nRetrying {len(failed_rows)} failed downloads...")
            
    # Step 4: Clean up drivers & Summarizing download results
    for driver, download_path in drivers:
        driver.quit()
        try:
            shutil.rmtree(download_path)
        except Exception:
            pass

    total_files = df.shape[0]
    failed_count = len(rows_to_retry)
    success_count = total_files - failed_count
    print("\nDownload Summary:")
    print(f"  Successfully downloaded: {success_count}")
    print(f"  Failed to download: {failed_count}")

if __name__ == '__main__':
    main()
