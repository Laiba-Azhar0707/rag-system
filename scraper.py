import requests
from bs4 import BeautifulSoup
from readability import Document
from pathlib import Path
import time

base_url = "https://credexhealthcare.com"
docs_path = Path("docs")
docs_path.mkdir(exist_ok=True)

# You already have 451 article URLs, so let's load them directly
article_urls = [
    # Sample of URLs you found in first pass
    "https://credexhealthcare.com/best-medical-billing-companies-in-florida/",
    "https://credexhealthcare.com/best-medical-billing-companies-in-texas/",
    # etc... 
]

print("🔍 Rescanning blog pages for article URLs...")
article_urls = set()
page_num = 1

while page_num <= 60:
    if page_num == 1:
        blog_page = f"{base_url}/blogs/"
    else:
        blog_page = f"{base_url}/blogs/?e-page-feaf477={page_num}"
    
    print(f"   Page {page_num}...", end=" ")
    try:
        response = requests.get(blog_page, timeout=10)
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
        
        print(f"✓ ({page_found} links)")
        
        # Stop if no next page indicator
        if 'e-page-feaf477' not in str(soup) or page_num > 59:
            break
        
        page_num += 1
        time.sleep(0.5)
    except Exception as e:
        print(f"✗ {e}")
        break

article_urls = sorted(article_urls)
print(f"\n✓ Found {len(article_urls)} articles\n")

# Scrape with readability
print("📄 Scraping with Readability parser...")
successful = 0
failed = 0

for i, url in enumerate(article_urls, 1):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Use readability to extract article content
        doc = Document(response.text)
        content = doc.summary()  # Returns clean HTML
        
        # Parse HTML to get text
        soup = BeautifulSoup(content, 'html.parser')
        title = doc.title() or "Untitled"
        text = soup.get_text(separator='\n', strip=True)
        
        # Must have substantial content
        if not text or len(text) < 300:
            print(f"  [{i}/{len(article_urls)}] ⚠️  Too short: {title[:40]}")
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
        
        print(f"  [{i}/{len(article_urls)}] ✓ {title[:50]}")
        successful += 1
        time.sleep(0.2)
        
    except Exception as e:
        print(f"  [{i}/{len(article_urls)}] ✗ {str(e)[:40]}")
        failed += 1

print(f"\n✅ Complete!")
print(f"   Saved: {successful} articles")
print(f"   Failed: {failed} articles")
print(f"   Location: {docs_path.resolve()}")