import streamlit as st
import feedparser
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="MarketPulse AI | Real-Time Sentiment", layout="wide", initial_sidebar_state="collapsed")

# Hide the sidebar completely to enforce button-only navigation
st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }
        .article-card { padding: 20px; border-radius: 10px; background-color: #1E1E1E; margin-bottom: 20px; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'news_data' not in st.session_state:
    st.session_state.news_data = []
if 'selected_article' not in st.session_state:
    st.session_state.selected_article = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = "Stock Market"

# ==========================================
# AI MODELS & CACHING
# ==========================================
@st.cache_resource(show_spinner="Loading AI Models (FinBERT & VADER)... This happens once.")
def load_models():
    # FinBERT for precise financial context
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    # VADER for lexicon-based heuristic scoring
    vader = SentimentIntensityAnalyzer()
    return tokenizer, model, vader

tokenizer, finbert_model, vader_analyzer = load_models()

# ==========================================
# HELPER FUNCTIONS (Scraping & NLP)
# ==========================================
def fetch_rss_news(query, limit=10):
    """Fetches news from Google News RSS feed using requests for safe URL encoding."""
    query = str(query).strip()
    url = "https://news.google.com/rss/search"
    params = {
        "q": f"{query} when:1d",
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en"
    }
    
    try:
        # Using requests to safely handle all URL encoding and special characters
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        # Parse the raw XML content safely
        feed = feedparser.parse(response.content)
    except Exception as e:
        return []
    
    articles = []
    for entry in feed.entries[:limit]:
        # Clean HTML from summary
        clean_summary = re.sub(r'<[^>]+>', '', entry.summary)
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": clean_summary,
            "source": entry.source.title if hasattr(entry, 'source') else "Google News"
        })
    return articles

def scrape_full_article(url):
    """Attempts to scrape the full text and main image of an article."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Try to find OG Image
        og_image = soup.find("meta", property="og:image")
        img_url = og_image["content"] if og_image else "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop"
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        content = "\n\n".join([p.text for p in paragraphs if len(p.text.split()) > 10])
        
        if len(content) < 200:
            content = "Full article content is protected by the publisher or requires javascript. Please click the original link to read the full text."
            
        return img_url, content
    except Exception as e:
        return "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop", "Could not load full article due to network or scraping restrictions."

def analyze_sentiment(text):
    """Runs text through FinBERT, VADER, and TextBlob."""
    # 1. FinBERT (Deep Learning)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    labels = ["Positive", "Negative", "Neutral"]
    prediction_idx = torch.argmax(predictions).item()
    finbert_label = labels[prediction_idx]
    finbert_confidence = predictions[0][prediction_idx].item()
    
    # 2. VADER (Lexicon)
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_compound = vader_scores['compound']
    
    # 3. TextBlob (Lexicon/Naive Bayes)
    blob_polarity = TextBlob(text).sentiment.polarity
    
    return {
        "FinBERT_Label": finbert_label,
        "FinBERT_Confidence": round(finbert_confidence * 100, 2),
        "VADER_Score": round(vader_compound, 3),
        "TextBlob_Score": round(blob_polarity, 3)
    }

# ==========================================
# NAVIGATION COMPONENT
# ==========================================
def render_navigation():
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    nav_items = ["Home", "Dashboard", "Scraped Articles", "About"]
    
    for i, col in enumerate(cols):
        # Highlight active page slightly differently if desired, here we just use standard buttons
        if col.button(f"🚀 {nav_items[i]}", key=f"nav_{nav_items[i]}", use_container_width=True):
            st.session_state.page = nav_items[i]
            st.rerun()
    st.markdown("---")

# ==========================================
# PAGE VIEWS
# ==========================================

def page_home():
    st.title("🤖 MarketPulse AI: The Intelligent Financial Analyst")
    st.image("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=2000&auto=format&fit=crop", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Project Overview")
        st.write("""
        MarketPulse AI is an industry-grade NLP pipeline designed to ingest live market news, process the linguistic nuances of financial reporting, and predict market sentiment in real-time. 
        It aggregates data to give traders and analysts a macro and micro view of market mood before price action occurs.
        """)
        
        st.header("The Problem Statement")
        st.write("""
        Retail and institutional investors face **Information Overload**. Thousands of news articles are published daily. By the time a human reads, comprehends, and contextualizes a breaking news story, algorithmic trading bots have already priced the news into the market. Manual sentiment analysis is too slow.
        """)
        
    with col2:
        st.header("The Solution")
        st.write("""
        An automated pipeline that scrapes RSS feeds and websites instantly, feeds the text into a fine-tuned Transformer model (**FinBERT**), and outputs a structured sentiment analysis. This allows users to view data-driven sentiment trends rather than relying on emotional interpretations of the news.
        """)
        
        st.header("How To Use It")
        st.info("""
        1. Navigate to the **Dashboard** using the buttons above.
        2. Enter a stock ticker (e.g., AAPL, TSLA) or a macro keyword (e.g., Interest Rates).
        3. Click 'Fetch & Analyze'. The AI will construct your dataset and graphs instantly.
        4. Go to **Scraped Articles** to read specific news and view detailed metrics.
        """)
        
    st.markdown("---")
    st.header("Why is this unique? (The Tech Edge)")
    st.write("""
    Standard sentiment analyzers (like standard VADER or TextBlob) fail at financial text. For example, the sentence *"Company X cuts its debt by 20%"* contains negative words ("cuts", "debt"), which standard AI flags as **Bearish**. 
    **MarketPulse AI uses FinBERT**, an AI trained specifically on financial documents, which correctly identifies debt reduction as a **Bullish** (Positive) signal. We combine this with Lexicon models to provide an ensemble confidence score.
    """)

def page_dashboard():
    st.title("📊 Real-Time Analytics Dashboard")
    
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        query = st.text_input("Enter Ticker or Topic (e.g., NVDA, Tech Stocks, Inflation):", value=st.session_state.search_query)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_clicked = st.button("Fetch & Analyze", type="primary", use_container_width=True)
        
    if fetch_clicked:
        st.session_state.search_query = query
        with st.spinner(f"Scraping web & running Deep Learning inference for '{query}'..."):
            articles = fetch_rss_news(query, limit=15)
            
            # Run inference on fetched articles
            for art in articles:
                # Analyze Title + Summary for better context
                combined_text = art['title'] + ". " + art['summary']
                sentiment_results = analyze_sentiment(combined_text)
                art.update(sentiment_results)
                
            st.session_state.news_data = articles
            
    # Display Dashboard if data exists
    if st.session_state.news_data:
        df = pd.DataFrame(st.session_state.news_data)
        
        st.markdown("### Executive Summary")
        total_arts = len(df)
        pos = len(df[df['FinBERT_Label'] == 'Positive'])
        neg = len(df[df['FinBERT_Label'] == 'Negative'])
        neu = len(df[df['FinBERT_Label'] == 'Neutral'])
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Articles Analyzed", total_arts)
        m2.metric("Bullish (Positive)", pos, f"{(pos/total_arts)*100:.0f}%")
        m3.metric("Bearish (Negative)", neg, f"-{(neg/total_arts)*100:.0f}%")
        m4.metric("Neutral", neu)
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### FinBERT Sentiment Distribution")
            fig_pie = px.pie(df, names='FinBERT_Label', 
                             color='FinBERT_Label',
                             color_discrete_map={'Positive':'#00FF41', 'Negative':'#FF003C', 'Neutral':'#FFAA00'},
                             hole=0.4)
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_chart2:
            st.markdown("#### VADER vs TextBlob (Lexicon Comparison)")
            # Create a bar chart showing VADER vs TextBlob for each article index
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=df.index, y=df['VADER_Score'], name='VADER (Social)', marker_color='#1f77b4'))
            fig_bar.add_trace(go.Bar(x=df.index, y=df['TextBlob_Score'], name='TextBlob (Naive)', marker_color='#ff7f0e'))
            fig_bar.update_layout(barmode='group', xaxis_title="Article Index", yaxis_title="Polarity Score", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### Scraped Data & Predictions (Tabular View)")
        display_df = df[['title', 'source', 'FinBERT_Label', 'FinBERT_Confidence', 'VADER_Score', 'published']]
        st.dataframe(display_df, use_container_width=True, height=300)
        
    else:
        st.info("No data available yet. Please enter a query and click 'Fetch & Analyze'.")

def page_articles():
    st.title("📰 Scraped Articles & Headlines")
    
    if not st.session_state.news_data:
        st.warning("No articles fetched yet. Please run an analysis in the Dashboard first.")
        if st.button("Go to Dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
        return

    st.write("Browse the latest scraped headlines. Click 'Read Full Article' to view details, extracted content, and deep sentiment analysis.")
    
    for idx, article in enumerate(st.session_state.news_data):
        with st.container():
            st.markdown(f"<div class='article-card'>", unsafe_allow_html=True)
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.subheader(article['title'])
                st.caption(f"📅 {article['published']} | 📰 Source: {article['source']}")
                
                # Dynamic coloring for sentiment
                color = "#00FF41" if article['FinBERT_Label'] == "Positive" else "#FF003C" if article['FinBERT_Label'] == "Negative" else "#FFAA00"
                st.markdown(f"**AI Prediction:** <span style='color:{color}'>{article['FinBERT_Label']}</span> (Confidence: {article['FinBERT_Confidence']}%)", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("Read Full Article", key=f"read_btn_{idx}", use_container_width=True):
                    st.session_state.selected_article = article
                    st.session_state.page = "Article View"
                    st.rerun()
                    
            st.markdown("</div>", unsafe_allow_html=True)

def page_article_view():
    article = st.session_state.selected_article
    
    if not article:
        st.error("No article selected.")
        if st.button("Back to Articles"):
            st.session_state.page = "Scraped Articles"
            st.rerun()
        return

    # Back button
    if st.button("⬅ Back to Headlines"):
        st.session_state.page = "Scraped Articles"
        st.rerun()
        
    st.markdown("---")
    
    with st.spinner("Dynamically scraping full article and image..."):
        img_url, full_content = scrape_full_article(article['link'])
    
    col_img, col_metrics = st.columns([2, 1])
    
    with col_img:
        st.image(img_url, use_container_width=True, caption=article['source'])
        
    with col_metrics:
        st.markdown("### AI Sentiment Analysis")
        color = "#00FF41" if article['FinBERT_Label'] == "Positive" else "#FF003C" if article['FinBERT_Label'] == "Negative" else "#FFAA00"
        st.markdown(f"## <span style='color:{color}'>{article['FinBERT_Label']}</span>", unsafe_allow_html=True)
        
        st.progress(article['FinBERT_Confidence'] / 100)
        st.caption(f"FinBERT Confidence: {article['FinBERT_Confidence']}%")
        
        st.markdown("**Alternative Metrics:**")
        st.write(f"- **VADER Score:** {article['VADER_Score']} (-1 to 1)")
        st.write(f"- **TextBlob Score:** {article['TextBlob_Score']} (-1 to 1)")
        
        st.markdown(f"[🔗 View Original Article on Publisher's Site]({article['link']})")

    st.title(article['title'])
    st.caption(f"Published: {article['published']}")
    st.markdown("### Extracted Content / Summary")
    st.write(full_content)
    
def page_about():
    st.title("👨‍💻 Developer Profile & Architecture")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=250)
        st.markdown("### Python Developer")
        st.write("**Focus:** AI/ML, Data Engineering, Web Apps")
        st.write("**Mission:** Bridging the gap between raw data and actionable financial intelligence.")
        
    with col2:
        st.header("Tech Stack Utilized")
        st.markdown("""
        * **Frontend & Routing:** `Streamlit` (Single Page Application architecture with custom Session State routing).
        * **Data Ingestion:** `feedparser` (RSS), `requests` & `BeautifulSoup4` (HTML Scraping).
        * **NLP Engine 1 (Deep Learning):** `Transformers` & `PyTorch` utilizing HuggingFace's **FinBERT**.
        * **NLP Engine 2 (Lexicon):** `vaderSentiment` & `TextBlob` for baseline comparisons.
        * **Data Processing & Viz:** `Pandas` for dataframe manipulation, `Plotly` for interactive, industry-grade charting.
        """)
        
        st.header("Why this Architecture?")
        st.write("""
        Using Streamlit allows for rapid deployment of Machine Learning models wrapped in a clean UI. By pushing the heavy NLP inference (FinBERT) into a cached function and running the scraping lazily, the app remains responsive. The dynamic HTML scraping acts as a failsafe to extract content even when standard APIs are unavailable.
        """)

# ==========================================
# MAIN APP ROUTER
# ==========================================
def main():
    # Always show top navigation
    render_navigation()
    
    # Route to the correct page based on session state
    if st.session_state.page == 'Home':
        page_home()
    elif st.session_state.page == 'Dashboard':
        page_dashboard()
    elif st.session_state.page == 'Scraped Articles':
        page_articles()
    elif st.session_state.page == 'Article View':
        page_article_view()
    elif st.session_state.page == 'About':
        page_about()

if __name__ == "__main__":
    main()
