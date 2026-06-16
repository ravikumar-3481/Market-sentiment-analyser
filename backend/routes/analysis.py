from fastapi import APIRouter, HTTPException, Query
from typing import List
from schemas import AnalyzeResponse
from news_scraper import fetch_rss_news
from sentiment_analyzer import analyze_sentiment, detect_topic

router = APIRouter()

@router.get("/api/analyze", response_model=List[AnalyzeResponse])
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
