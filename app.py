import streamlit as st
import feedparser
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import re
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="MarketPulse AI | Real-Time Sentiment", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }
        .article-card { padding: 20px; border-radius: 10px; background-color: #1E1E1E; margin-bottom: 20px; border: 1px solid #333; }
        .entity-badge { padding: 3px 8px; border-radius: 5px; font-size: 0.85em; font-weight: bold; margin-right: 5px; color: black; }
        .entity-ORG { background-color: #a6e22e; }
        .entity-PER { background-color: #fd971f; }
        .entity-LOC { background-color: #66d9ef; }
        .entity-MISC { background-color: #f92672; }
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
    st.session_state.search_query = "NVDA"

# ==========================================
# AI MODELS (LAZY LOADING FOR MEMORY SAFETY)
# ==========================================
# We load models separately to prevent Streamlit Cloud from crashing due to Out-Of-Memory (OOM) errors.

@st.cache_resource(show_spinner="Loading FinBERT Sentiment Engine...")
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

@st.cache_resource(show_spinner="Loading Lexicon Engine...")
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner="Loading AI Topic Classifier...")
def load_topic_model():
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

@st.cache_resource(show_spinner="Loading AI Summarizer...")
def load_summarizer():
    # Using Falconsai as it is highly efficient and won't crash Streamlit's 1GB RAM limit
    return pipeline("summarization", model="Falconsai/text_summarization")

@st.cache_resource(show_spinner="Loading Entity Extractor...")
def load_ner():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")


# ==========================================
# HELPER FUNCTIONS
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
    """Scrapes Reddit RSS (wallstreetbets/stocks) securely."""
    url = f"https://www.reddit.com/r/wallstreetbets/search.rss?q={quote(query)}&restrict_sr=on&sort=new&t=week"
    headers = {'User-Agent': 'MarketPulseAI/1.0 (Educational Bot)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        feed = feedparser.parse(response.content)
        posts = []
        for entry in feed.entries[:limit]:
            posts.append({"title": entry.title, "published": entry.published})
        return posts
    except:
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
    tokenizer, finbert_model = load_finbert()
    vader_analyzer = load_vader()
    
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
    topic_pipe = load_topic_model()
    candidate_labels = ["Earnings", "Mergers & Acquisitions", "Macroeconomics", "Leadership/Management", "Regulatory/Legal"]
    result = topic_pipe(text, candidate_labels)
    return result['labels'][0]

# ==========================================
# NAVIGATION
# ==========================================
def render_navigation():
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(5)
    nav_items = ["Home", "Dashboard", "Market Data", "Scraped Articles", "About"]
    
    for i, col in enumerate(cols):
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
        st.write("MarketPulse AI is an industry-grade NLP pipeline designed to ingest live market news, process linguistic nuances using advanced ML models, and predict market sentiment in real-time.")
        st.header("Advanced Capabilities")
        st.write("""
        - **Deep Learning Sentiment:** Powered by FinBERT.
        - **Zero-Shot Topic Modeling:** Automatically categorizes articles.
        - **AI Summarization:** Condenses long financial reports instantly.
        - **Named Entity Recognition (NER):** Extracts key companies and figures.
        - **Social vs. Institutional Data:** Compares Reddit sentiment against Wall Street news.
        """)
    with col2:
        st.header("How To Use It")
        st.info("""
        1. Navigate to **Dashboard** for News Sentiment & Topic analysis.
        2. Navigate to **Market Data** to compare real-time Stock Prices with Reddit chatter.
        3. Go to **Scraped Articles** to use the AI Summarizer and Entity Extraction tools.
        """)

def page_dashboard():
    st.title("📊 Financial News & Topic Dashboard")
    
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        query = st.text_input("Enter Ticker or Topic", value=st.session_state.search_query)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Fetch & Analyze", type="primary", use_container_width=True):
            st.session_state.search_query = query
            with st.spinner(f"Scraping & running AI Pipelines for '{query}'..."):
                articles = fetch_rss_news(query, limit=20)
                for art in articles:
                    combined_text = art['title'] + ". " + art['summary']
                    sentiment_results = analyze_sentiment(combined_text)
                    art.update(sentiment_results)
                    # Run Topic Modeling
                    art['Topic'] = detect_topic(art['title'])
                st.session_state.news_data = articles
            st.rerun()
            
    if st.session_state.news_data:
        df = pd.DataFrame(st.session_state.news_data)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Articles Analyzed", len(df))
        m2.metric("Bullish (Positive)", len(df[df['FinBERT_Label'] == 'Positive']))
        m3.metric("Bearish (Negative)", len(df[df['FinBERT_Label'] == 'Negative']))
        m4.metric("Neutral", len(df[df['FinBERT_Label'] == 'Neutral']))
        
        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### Sentiment Distribution")
            fig_pie = px.pie(df, names='FinBERT_Label', color='FinBERT_Label',
                             color_discrete_map={'Positive':'#00FF41', 'Negative':'#FF003C', 'Neutral':'#FFAA00'}, hole=0.4)
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_chart2:
            st.markdown("#### AI Topic Modeling (Zero-Shot)")
            topic_counts = df['Topic'].value_counts().reset_index()
            topic_counts.columns = ['Topic', 'Count']
            fig_bar = px.bar(topic_counts, x='Topic', y='Count', color='Topic')
            fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### Processed Intelligence Data")
        st.dataframe(df[['title', 'Topic', 'FinBERT_Label', 'FinBERT_Confidence']], use_container_width=True)

def page_market_data():
    st.title("📈 Market Data & Social Sentiment")
    st.write("Compare historical price action against volatile social media sentiment (Reddit r/WallStreetBets).")
    
    ticker = st.text_input("Enter exact Stock Ticker (e.g., AAPL, TSLA, NVDA):", value=st.session_state.search_query.split()[0].upper())
    
    if st.button("Load Market Data"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### {ticker} Price Action (3 Months)")
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="3mo")
                if not hist.empty:
                    fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price")])
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(window=20).mean(), line=dict(color='orange', width=2), name='20-Day MA'))
                    fig.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Invalid ticker or no data found on Yahoo Finance.")
            except Exception as e:
                st.error(f"Error loading stock data: {e}")
                
        with col2:
            st.markdown("### Reddit Live Pulse")
            with st.spinner("Scraping Reddit..."):
                reddit_posts = fetch_reddit_sentiment(ticker)
            
            if reddit_posts:
                vader_analyzer = load_vader()
                pos_count = 0
                neg_count = 0
                for post in reddit_posts:
                    vs = vader_analyzer.polarity_scores(post['title'])['compound']
                    if vs > 0.05: pos_count += 1
                    elif vs < -0.05: neg_count += 1
                
                st.metric("Reddit Posts Analyzed", len(reddit_posts))
                st.progress(pos_count / max(1, (pos_count + neg_count)))
                st.caption("Ratio of Bullish vs Bearish social posts.")
                
                st.markdown("**Latest Social Chatter:**")
                for p in reddit_posts[:5]:
                    st.markdown(f"- *{p['title']}*")
            else:
                st.warning("No recent Reddit data found for this query.")

def page_articles():
    st.title("📰 Scraped Articles & Deep Analysis")
    if not st.session_state.news_data:
        st.warning("Run an analysis in the Dashboard first.")
        return
    
    for idx, article in enumerate(st.session_state.news_data):
        with st.container():
            st.markdown(f"<div class='article-card'>", unsafe_allow_html=True)
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(article['title'])
                st.caption(f"Topic: {article.get('Topic', 'Unknown')} | Source: {article['source']}")
                color = "#00FF41" if article['FinBERT_Label'] == "Positive" else "#FF003C" if article['FinBERT_Label'] == "Negative" else "#FFAA00"
                st.markdown(f"**AI Prediction:** <span style='color:{color}'>{article['FinBERT_Label']}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Deep Dive Analysis", key=f"read_{idx}", use_container_width=True):
                    st.session_state.selected_article = article
                    st.session_state.page = "Article View"
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

def page_article_view():
    article = st.session_state.selected_article
    if not article:
        st.error("No article selected.")
        return
        
    if st.button("⬅ Back to Headlines"):
        st.session_state.page = "Scraped Articles"
        st.rerun()
        
    st.markdown("---")
    with st.spinner("Scraping full text..."):
        img_url, full_content = scrape_full_article(article['link'])
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title(article['title'])
        st.image(img_url, use_container_width=True)
        st.markdown(f"[🔗 View Original]({article['link']})")
        st.write(full_content)
        
    with col2:
        st.markdown("### AI Summarization (TL;DR)")
        if st.button("Generate Summary"):
            with st.spinner("Neural network is reading the text..."):
                if len(full_content) > 300:
                    summarizer_pipe = load_summarizer()
                    summary = summarizer_pipe(full_content[:2000], max_length=130, min_length=30, do_sample=False)
                    st.success(summary[0]['summary_text'])
                else:
                    st.warning("Text too short to summarize.")
        
        st.markdown("---")
        st.markdown("### Named Entity Recognition (NER)")
        if st.button("Extract Entities"):
            with st.spinner("Extracting..."):
                ner_pipe = load_ner()
                entities = ner_pipe(full_content[:2000])
                # Deduplicate and format entities
                unique_entities = {"ORG": set(), "PER": set(), "LOC": set(), "MISC": set()}
                for ent in entities:
                    tag = ent['entity_group']
                    unique_entities[tag].add(ent['word'].replace("##", ""))
                
                for tag, words in unique_entities.items():
                    if words:
                        html = f"<br><b>{tag}:</b> "
                        for w in words:
                            html += f"<span class='entity-badge entity-{tag}'>{w}</span>"
                        st.markdown(html, unsafe_allow_html=True)

def page_about():
    st.title("👨‍💻 Developer Profile & Architecture")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://profileravi.netlify.app/og/img.webp", width=250)
        st.markdown("### Ravi Vishwakarma")
        st.write("**Focus:** AI/ML, Data Engineering, Web Apps")
    with col2:
        st.header("Enterprise Architecture")
        st.write("""
        This app uses a multi-model ensemble approach:
        - **Data Pipelines:** `feedparser`, `yfinance`, `beautifulsoup4`.
        - **NLP Transformers:** HuggingFace `pipeline` routing 4 distinct neural networks (FinBERT for sentiment, Falconsai for Summarization, BERT for NER, DistilBERT-MNLI for Topic classification).
        - **Data Viz:** `plotly` & `pandas`.
        - **Memory Optimization:** Implements aggressive Lazy-Loading via discrete `@st.cache_resource` states to run efficiently on cloud environments.
        """)

def main():
    render_navigation()
    pages = {
        'Home': page_home,
        'Dashboard': page_dashboard,
        'Market Data': page_market_data,
        'Scraped Articles': page_articles,
        'Article View': page_article_view,
        'About': page_about
    }
    pages.get(st.session_state.page, page_home)()

if __name__ == "__main__":
    main()
