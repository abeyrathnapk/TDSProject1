import requests
import json
import time
import os

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_PATH = "/c/courses/tds-kb/34"
OUTPUT_FILE = "tds_course_content.json"  # Use the same file as the course content

# Paste your _t cookie value here
SESSION_COOKIE = "op0dfzlDLILy46rRFEtzzH5wi%2FZHP9g%2Bp6EOK8jxzyiT22Z2zQkAWdBc7jWyWENAW7EWTxfE0TYddCcm2ZYRok2aD1MhsLjgQ3O6tRF55MBOahxPQXqI5kPEEL6ctNlx6g44H8tvZ6Pn%2FUTrs%2F1IEjKKZo%2FOr%2F7EI0b72yFeegWUrWULIOS%2FaoFRu16%2FAGn%2BgarAAKiZOSLGWifiilTjjX0vD6nO78V6%2FjHvWzKydQDi5RZFmMjL%2Fo3G77zab%2FpJjbatycPsydLTLsucySlBA%2BxrplEGDCSGXq2xgwgkuqZWFi%2FUPLlV6xRE5O6Qlpi4--uN11JuFn%2FIygUiUj--gxrgNEdvUoMgERDq%2FuN4hQ%3D%3D"

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
}
COOKIES = {
    '_t': SESSION_COOKIE
}


def get_topics():
    topics = []
    page = 0
    while True:
        url = f"{BASE_URL}{CATEGORY_PATH}.json?page={page}"
        print(f"Fetching topics from: {url}")
        resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
        if resp.status_code != 200:
            break
        data = resp.json()
        topic_list = data.get('topic_list', {}).get('topics', [])
        if not topic_list:
            break
        topics.extend(topic_list)
        page += 1
        time.sleep(1)
    return topics


def get_posts_for_topic(topic_id):
    url = f"{BASE_URL}/t/{topic_id}.json"
    print(f"Fetching posts for topic: {url}")
    resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
    if resp.status_code != 200:
        return []
    data = resp.json()
    posts = data.get('post_stream', {}).get('posts', [])
    topic_title = data.get('title', '')
    topic_slug = data.get('slug', '')
    topic_url = f"{BASE_URL}/t/{topic_slug}/{topic_id}"
    result = []
    for post in posts:
        result.append({
            'source': 'discourse',
            'topic_id': topic_id,
            'topic_title': topic_title,
            'topic_url': topic_url,
            'post_number': post.get('post_number'),
            'author': post.get('username'),
            'created_at': post.get('created_at'),
            'content': post.get('cooked')  # HTML content
        })
    return result


def main():
    # Load existing course content if present
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    else:
        all_data = []

    topics = get_topics()
    print(f"Found {len(topics)} topics.")
    for topic in topics:
        # Only process topics from the exact tds-kb category (category_id == 34)
        if topic.get('category_id') != 34:
            continue
        topic_id = topic['id']
        posts = get_posts_for_topic(topic_id)
        all_data.extend(posts)
        # Save progress after each topic
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        time.sleep(1)
    print(f"Saved combined data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
