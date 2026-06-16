import re
import requests
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import quote

def fetch_rss_news(query, limit=15):
    query = str(query).strip()
    url = "https://news.google.com/rss/search"
    params = {"q": f"{query} when:1d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception as e:
        print(f"Error fetching RSS news: {e}")
        return []
    
    articles = []
    for entry in feed.entries[:limit]:
        clean_summary = re.sub(r'<[^>]+>', '', entry.summary)
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": clean_summary,
            "source": entry.source.title if hasattr(entry, 'source') else "Google News"
        })
    return articles

def fetch_reddit_sentiment(query, limit=15):
    url = f"https://www.reddit.com/r/wallstreetbets/search.rss?q={quote(query)}&restrict_sr=on&sort=new&t=week"
    headers = {'User-Agent': 'MarketPulseAI/1.0 (Educational Bot)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        feed = feedparser.parse(response.content)
        posts = []
        for entry in feed.entries[:limit]:
            posts.append({"title": entry.title, "published": entry.published})
        return posts
    except Exception as e:
        print(f"Error fetching Reddit sentiment: {e}")
        return []

def scrape_full_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        og_image = soup.find("meta", property="og:image")
        img_url = og_image["content"] if og_image else "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop"
        paragraphs = soup.find_all('p')
        content = "\n\n".join([p.text for p in paragraphs if len(p.text.split()) > 10])
        if len(content) < 200:
            content = "Full article content is protected. Please click the original link to read."
        return img_url, content
    except Exception:
        return "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop", "Could not load full article."
