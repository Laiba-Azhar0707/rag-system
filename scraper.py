import requests
from bs4 import BeautifulSoup
from readability import Document
from pathlib import Path
from urllib.parse import urljoin
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

base_url = "https://credexhealthcare.com"
docs_path = Path("docs")
docs_path.mkdir(exist_ok=True)

logger.info("Starting blog page scan...")

# Fetch blog listing pages and collect all article URLs
article_urls = set()
page_num = 1

while page_num <= 60:
    if page_num == 1:
        blog_page = f"{base_url}/blogs/"
    else:
        blog_page = f"{base_url}/blogs/?e-page-feaf477={page_num}"
    
    try:
        logger.info(f"Scanning page {page_num}...")
        response = requests.get(blog_page, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        page_found = 0
        # Extract all links with article patterns
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(kw in href.lower() for kw in ['/best-', '/medical-', '/how-', '/credentialing', '/billing', '/provider']):
                if href.startswith('/'):
                    href = base_url + href
                if base_url in href and '#' not in href:
                    article_urls.add(href)
                    page_found += 1
        
        logger.info(f"  Page {page_num}: Found {page_found} links")
        
        # Stop if no next page indicator
        if 'e-page-feaf477' not in str(soup) or page_num > 59:
            break
        
        page_num += 1
        time.sleep(0.5)
        
    except requests.RequestException as e:
        logger.error(f"Request failed on page {page_num}: {e}")
        break
    except Exception as e:
        logger.error(f"Unexpected error on page {page_num}: {e}")
        break

article_urls = sorted(article_urls)
logger.info(f"Found {len(article_urls)} unique articles\n")

# Scrape each article
logger.info("Starting article scraping...")
successful = 0
failed = 0

for i, url in enumerate(article_urls, 1):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Use readability to extract article content
        doc = Document(response.text)
        
        if not doc or not doc.summary():
            logger.warning(f"[{i}/{len(article_urls)}] Empty content: {url[:60]}")
            failed += 1
            continue
        
        content = doc.summary()
        
        # Parse HTML to get text
        soup = BeautifulSoup(content, 'html.parser')
        title = doc.title() or "Untitled"
        text = soup.get_text(separator='\n', strip=True)
        
        # Must have substantial content
        if not text or len(text) < 300:
            logger.warning(f"[{i}/{len(article_urls)}] Too short ({len(text)} chars): {title[:40]}")
            failed += 1
            continue
        
        # Save file
        filename = title.lower()
        filename = ''.join(c if c.isalnum() or c == ' ' else '_' for c in filename)
        filename = '_'.join(filename.split())[:60] + '.txt'
        
        filepath = docs_path / filename
        counter = 1
        while filepath.exists():
            base = filename.rsplit('.', 1)[0]
            filename = f"{base}_{counter}.txt"
            filepath = docs_path / filename
            counter += 1
        
        filepath.write_text(text, encoding='utf-8')
        
        logger.info(f"[{i}/{len(article_urls)}] ✓ {title[:50]}")
        successful += 1
        time.sleep(0.2)
        
    except requests.Timeout:
        logger.error(f"[{i}/{len(article_urls)}] Timeout: {url[:60]}")
        failed += 1
    except requests.RequestException as e:
        logger.error(f"[{i}/{len(article_urls)}] Request failed: {str(e)[:50]}")
        failed += 1
    except Exception as e:
        logger.error(f"[{i}/{len(article_urls)}] Parsing error: {str(e)[:50]}")
        failed += 1

logger.info(f"\n{'='*50}")
logger.info(f"✅ Complete!")
logger.info(f"   Saved: {successful} articles")
logger.info(f"   Failed: {failed} articles")
logger.info(f"   Location: {docs_path.resolve()}")
logger.info(f"{'='*50}")