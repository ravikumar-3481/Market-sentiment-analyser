import feedparser
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from datetime import datetime
from urllib.parse import quote
import numpy as np
import pandas as pd
import yfinance as yf
import re
import warnings
import torch

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="MarketPulse AI API", description="Enterprise Financial Intelligence API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ML MODELS LAZY LOADING
# ==========================================
finbert_tokenizer = None
finbert_model = None
vader_analyzer = None
topic_pipeline = None
summarize_pipeline = None
ner_pipeline = None

def get_finbert():
    global finbert_tokenizer, finbert_model
    if finbert_tokenizer is None or finbert_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("Loading FinBERT Sentiment Engine...")
        finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return finbert_tokenizer, finbert_model

def get_vader():
    global vader_analyzer
    if vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("Loading VADER Lexicon Engine...")
        vader_analyzer = SentimentIntensityAnalyzer()
    return vader_analyzer

def get_topic_model():
    global topic_pipeline
    if topic_pipeline is None:
        from transformers import pipeline
        print("Loading Zero-Shot Topic Classifier...")
        topic_pipeline = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    return topic_pipeline

def get_summarizer():
    global summarize_pipeline
    if summarize_pipeline is None:
        from transformers import pipeline
        print("Loading AI Summarizer...")
        summarize_pipeline = pipeline("summarization", model="Falconsai/text_summarization")
    return summarize_pipeline

def get_ner():
    global ner_pipeline
    if ner_pipeline is None:
        from transformers import pipeline
        print("Loading Entity Extractor...")
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return ner_pipeline


# ==========================================
# CORE PROCESSING UTILITIES
# ==========================================
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

def analyze_sentiment(text):
    tokenizer, finbert_model = get_finbert()
    vader_analyzer = get_vader()
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["Positive", "Negative", "Neutral"]
    prediction_idx = torch.argmax(predictions).item()
    return {
        "FinBERT_Label": labels[prediction_idx],
        "FinBERT_Confidence": round(predictions[0][prediction_idx].item() * 100, 2),
        "VADER_Score": round(vader_analyzer.polarity_scores(text)['compound'], 3)
    }

def detect_topic(text):
    topic_pipe = get_topic_model()
    candidate_labels = ["Earnings", "Mergers & Acquisitions", "Macroeconomics", "Leadership/Management", "Regulatory/Legal", "Product Launch"]
    result = topic_pipe(text, candidate_labels)
    return result['labels'][0]


# ==========================================
# API ENDPOINTS
# ==========================================

class AnalyzeResponse(BaseModel):
    title: str
    link: str
    published: str
    summary: str
    source: str
    FinBERT_Label: str
    FinBERT_Confidence: float
    VADER_Score: float
    Topic: str

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/analyze", response_model=List[AnalyzeResponse])
def analyze_query(query: str = Query(..., min_length=1), limit: int = 15):
    try:
        articles = fetch_rss_news(query, limit=limit)
        if not articles:
            return []
        
        results = []
        for art in articles:
            combined_text = art['title'] + ". " + art['summary']
            sentiment = analyze_sentiment(combined_text)
            topic = detect_topic(art['title'])
            
            results.append({
                **art,
                "FinBERT_Label": sentiment["FinBERT_Label"],
                "FinBERT_Confidence": sentiment["FinBERT_Confidence"],
                "VADER_Score": sentiment["VADER_Score"],
                "Topic": topic
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ScrapeRequest(BaseModel):
    url: str

class ScrapeResponse(BaseModel):
    image_url: str
    content: str

@app.post("/api/scrape-article", response_model=ScrapeResponse)
def scrape_article(request: ScrapeRequest):
    try:
        image_url, content = scrape_full_article(request.url)
        return {"image_url": image_url, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ContentRequest(BaseModel):
    content: str

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize_content(request: ContentRequest):
    try:
        if len(request.content) < 200:
            return {"summary": "Content is too short to summarize."}
        
        summarizer_pipe = get_summarizer()
        # Cap input length for model safety
        truncated_content = request.content[:2000]
        summary = summarizer_pipe(truncated_content, max_length=130, min_length=30, do_sample=False)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EntityResponse(BaseModel):
    ORG: List[str]
    PER: List[str]
    LOC: List[str]
    MISC: List[str]

@app.post("/api/extract-entities", response_model=EntityResponse)
def extract_entities(request: ContentRequest):
    try:
        ner_pipe = get_ner()
        truncated_content = request.content[:2000]
        entities = ner_pipe(truncated_content)
        
        unique_entities = {"ORG": set(), "PER": set(), "LOC": set(), "MISC": set()}
        for ent in entities:
            tag = ent['entity_group']
            if tag in unique_entities:
                word = ent['word'].replace("##", "").strip()
                if len(word) > 1:
                    unique_entities[tag].add(word)
        
        return {
            "ORG": sorted(list(unique_entities["ORG"])),
            "PER": sorted(list(unique_entities["PER"])),
            "LOC": sorted(list(unique_entities["LOC"])),
            "MISC": sorted(list(unique_entities["MISC"]))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MarketDataResponse(BaseModel):
    ticker: str
    metrics: dict
    historical_data: List[dict]
    reddit_sentiment: dict

@app.get("/api/market-data", response_model=MarketDataResponse)
def get_market_data(ticker: str = Query(..., min_length=1)):
    ticker = ticker.upper().strip()
    
    # Fetch stock price action
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yahoo Finance connection failed: {str(e)}")
        
    metrics = {}
    historical_list = []
    
    if not hist.empty:
        latest_close = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else latest_close
        pct_change = float(((latest_close - prev_close) / prev_close) * 100)
        period_high = float(hist['High'].max())
        period_low = float(hist['Low'].min())
        
        metrics = {
            "latest_close": round(latest_close, 2),
            "prev_close": round(prev_close, 2),
            "pct_change": round(pct_change, 2),
            "period_high": round(period_high, 2),
            "period_low": round(period_low, 2)
        }
        
        # Build candlestick data list
        display_hist = hist.copy()
        display_hist.reset_index(inplace=True)
        date_col = 'Date' if 'Date' in display_hist.columns else 'Datetime'
        
        display_hist[date_col] = display_hist[date_col].dt.strftime('%Y-%m-%d')
        
        # Calculate moving average
        display_hist['MA5'] = display_hist['Close'].rolling(window=5).mean()
        
        for _, row in display_hist.iterrows():
            historical_list.append({
                "Date": row[date_col],
                "Open": round(float(row['Open']), 2),
                "High": round(float(row['High']), 2),
                "Low": round(float(row['Low']), 2),
                "Close": round(float(row['Close']), 2),
                "Volume": int(row['Volume']),
                "MA5": round(float(row['MA5']), 2) if not pd.isna(row['MA5']) else None
            })
            
        # Reverse to show newest historical entries first for table, but keep chronological order for charts
        historical_list.sort(key=lambda x: x["Date"])
    else:
        # Ticker might be invalid or no data returned
        raise HTTPException(status_code=400, detail="Invalid ticker or no recent trading data found.")
        
    # Fetch Reddit Chatter
    reddit_posts = fetch_reddit_sentiment(ticker)
    reddit_data = {
        "total_mentions": 0,
        "bullish_ratio": 0.0,
        "pos_count": 0,
        "neg_count": 0,
        "neu_count": 0,
        "posts": []
    }
    
    if reddit_posts:
        vader = get_vader()
        pos_count, neg_count, neu_count = 0, 0, 0
        posts_list = []
        
        for post in reddit_posts:
            title = post['title']
            published = post['published']
            vs = vader.polarity_scores(title)['compound']
            
            if vs > 0.05:
                label = "Bullish"
                pos_count += 1
            elif vs < -0.05:
                label = "Bearish"
                neg_count += 1
            else:
                label = "Neutral"
                neu_count += 1
                
            posts_list.append({
                "title": title,
                "published": published,
                "sentiment": label,
                "score": round(vs, 3)
            })
            
        total = len(reddit_posts)
        bull_ratio = pos_count / max(1, (pos_count + neg_count))
        
        reddit_data = {
            "total_mentions": total,
            "bullish_ratio": round(bull_ratio, 2),
            "pos_count": pos_count,
            "neg_count": neg_count,
            "neu_count": neu_count,
            "posts": posts_list
        }
        
    return {
        "ticker": ticker,
        "metrics": metrics,
        "historical_data": historical_list,
        "reddit_sentiment": reddit_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
