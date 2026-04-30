import requests
import os
import json
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time

BASE_DIR = Path(__file__).parents[0]
CSV_PATH = BASE_DIR / 'data' / 'websites' / 'sklad_mebliv' / 'skladmebliv_bedrooms.csv'
SAVE_PATH = BASE_DIR / 'data' / 'websites' / 'sklad_mebliv'


def get_furniture_info(session, furniture_url):
    info = []
    try:
        response = session.get(furniture_url)
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f'Failed scrape furniture item on {furniture_url} due to {e}')
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')

    img_tag = soup.select_one("a.modal-popup img")
    if img_tag:
        img_url = img_tag["src"]

        info.append(img_url)
    else:
        print(f"No image found for {furniture_url}")
        info.append(None)
    category = soup.find("ul", class_="breadcrumb")
    if category:
        category_items = category.find_all("li")
        category_name_1 = ""
        category_name_2 = ""
        if len(category_items) >= 4:
            category_name_1 = category_items[3].get_text(strip=True)
        if len(category_items) >= 3:
            category_name_2 = category_items[2].get_text(strip=True)
        category_text = f"{category_name_1} | {category_name_2}".strip(" | ")
        info.append(category_text if category_text else None)
    else:
        print(f"No category found for {furniture_url}")
        info.append(None)

    return info

def scrape_scene_page(session, scene_url):
    try:
        response = session.get(scene_url)
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f'Failed scrape scene page on {scene_url} due to {e}')
        return {}
    
    soup = BeautifulSoup(response.text, 'html.parser')
    furniture_items = soup.select("div.component.listing-data")
    furniture_data = []
    for item in furniture_items:
        # name
        name = item.select_one("a.component__name span").get_text(strip=True)

        # href
        href = item.select_one("a.component__name")["href"]

        # image = item.select_one("img.component__image-main")["data-src"]

        furniture_info = get_furniture_info(session, href)
        furniture_image = furniture_info[0] if len(furniture_info) > 0 and furniture_info[0] else ""
        category = furniture_info[1] if len(furniture_info) > 1 and furniture_info[1] else ""

        furniture_data.append({
            "name": name,
            "href": href,
            "image": furniture_image,
            "category": category
        })
        time.sleep(2)
    
    return furniture_data

def get_random_headers():
    ua = UserAgent(browsers=['Edge', 'Chrome', 'Firefox', 'Google', 'Opera'],
               platforms='desktop')

    headers = {"User-Agent": ua.random,
    'Accept': (
        'text/html,application/xhtml+xml,application/xml;'
        'q=0.9,image/avif,image/webp,image/apng,*/*;'
        'q=0.8,application/signed-exchange;v=b3;q=0.7'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
    }
    return headers

def main():
    save_file = os.path.join(SAVE_PATH, 'skladmebliv_bedrooms_parsed.json')
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = []
        for i, row in enumerate(reader):
            scene_name = row['product__name']
            scene_href = row['product__image href']
            scene_img_src = row['product__image-main src']

            try:
                session = requests.Session()
                session.headers = get_random_headers()
                response = session.get(scene_href)
                response.raise_for_status()
            except requests.HTTPError as e:
                print(f'Failed scrape {scene_href} due to {e}')
                continue
            
            furnitures = scrape_scene_page(session, scene_href)


            data.append({
                "scene_name": scene_name,
                "scene_href": scene_href,
                "scene_img_src": scene_img_src,
                "furnitures_items": furnitures
            })
            
            print(f'Processing {i} scene: {scene_name} with {len(furnitures)} furniture items')
            time.sleep(3)
    
    
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()