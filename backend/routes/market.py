from fastapi import APIRouter, HTTPException, Query
import yfinance as yf
import pandas as pd
from schemas import MarketDataResponse
from news_scraper import fetch_reddit_sentiment
from ml_loader import get_vader

router = APIRouter()

@router.get("/api/market-data", response_model=MarketDataResponse)
def get_market_data(ticker: str = Query(..., min_length=1)):
    ticker = ticker.upper().strip()
    
    # Fetch stock price action
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if not hist.empty:
            hist = hist.dropna(subset=['Open', 'High', 'Low', 'Close'])
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
