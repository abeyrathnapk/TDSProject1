# TDS Course Scraper using Selenium
# Requirements:
#   pip install selenium
#   Download ChromeDriver from https://sites.google.com/chromium.org/driver/ and place it in your PATH

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
import time

BASE_URL = "https://tds.s-anand.net/"
OUTPUT_FILE = "tds_course_content.json"


def get_internal_links(driver):
    # Wait for page to load
    time.sleep(2)
    links = set()
    a_tags = driver.find_elements(By.TAG_NAME, 'a')
    for a in a_tags:
        href = a.get_attribute('href')
        if href and href.startswith(BASE_URL + "#/"):
            # Only keep the hash part
            hash_part = href.split(BASE_URL)[-1]
            links.add(hash_part)
    return list(links)


def scrape_page(driver, url):
    driver.get(url)
    time.sleep(2)  # Wait for JS to render
    title = driver.title
    # Get main visible text (customize selector as needed)
    body = driver.find_element(By.TAG_NAME, 'body')
    main_text = body.text
    return {
        'url': url,
        'title': title,
        'content': main_text
    }


def main():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=chrome_options)

    print(f"Scraping main page: {BASE_URL}")
    driver.get(BASE_URL)
    links = get_internal_links(driver)
    print(f"Found {len(links)} internal links.")

    all_data = []
    # Scrape main page
    all_data.append(scrape_page(driver, BASE_URL))

    for link in links:
        full_url = BASE_URL + link
        print(f"Scraping: {full_url}")
        try:
            data = scrape_page(driver, full_url)
            all_data.append(data)
            time.sleep(1)
        except Exception as e:
            print(f"Failed to scrape {full_url}: {e}")

    driver.quit()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"Saved scraped data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
