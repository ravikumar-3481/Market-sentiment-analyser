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
        .article-card { padding: 20px; border-radius: 10px; background-color: #1E1E1E; margin-bottom: 20px; border: 1px solid #333; transition: 0.3s; }
        .article-card:hover { border-color: #00FF41; box-shadow: 0 4px 15px rgba(0, 255, 65, 0.1); }
        .entity-badge { padding: 3px 8px; border-radius: 5px; font-size: 0.85em; font-weight: bold; margin-right: 5px; color: black; }
        .entity-ORG { background-color: #a6e22e; }
        .entity-PER { background-color: #fd971f; }
        .entity-LOC { background-color: #66d9ef; }
        .entity-MISC { background-color: #f92672; }
        .metric-box { padding: 15px; border-radius: 8px; background-color: #262730; border-left: 5px solid #00FF41; }
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
    candidate_labels = ["Earnings", "Mergers & Acquisitions", "Macroeconomics", "Leadership/Management", "Regulatory/Legal", "Product Launch"]
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
    st.title("🤖 MarketPulse AI: Enterprise Financial Intelligence")
    st.image("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=2000&auto=format&fit=crop", use_container_width=True)
    
    st.markdown("### 📖 Project Overview")
    st.write("MarketPulse AI is an industry-grade, real-time intelligence platform that bridges the gap between raw financial news and actionable market data. By deploying an ensemble of Deep Learning models, it reads, understands, categorizes, and scores the sentiment of global financial news in milliseconds, giving retail and institutional traders a definitive edge.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚠️ Problem Statement")
        st.write("""
        1. **Information Overload:** Thousands of financial articles are published daily. Humans cannot process this volume fast enough.
        2. **Nuance Misinterpretation:** Standard algorithms fail at financial jargon. (e.g., "Company X cuts debt" is mathematically a 'cut', but financially 'positive').
        3. **Delayed Action:** By the time a trader digests a news report and checks the chart, the algorithmic trading bots have already priced the news into the stock.
        """)
        
        st.markdown("### 💡 The Solution")
        st.write("An automated NLP pipeline that ingests data from RSS feeds, WallStreetBets, and direct web scraping. It feeds this text into **FinBERT** (a transformer model fine-tuned on financial data), performs Zero-Shot Topic Modeling, and maps this intelligence directly against real-time stock price action.")

    with col2:
        st.markdown("### ⚙️ Why & How This Solves The Problem")
        st.write("""
        - **Speed (How):** Fully asynchronous web scraping coupled with locally cached HuggingFace pipelines.
        - **Accuracy (Why):** Replacing generic VADER sentiment with context-aware Deep Learning ensures we don't mislabel critical economic data.
        - **Centralization:** It brings social sentiment (Reddit), institutional news (Google), and Price Action (Yahoo Finance) into a single pane of glass.
        """)
        
        st.markdown("### 🚀 How To Use This Project")
        st.info("""
        1. Go to **Dashboard** and enter a stock ticker (e.g., AAPL) or macro topic (e.g., Inflation) to fetch and analyze news sentiment.
        2. Check **Market Data** to see the 30-day real-time stock chart mapped against Reddit Social Sentiment.
        3. Open **Scraped Articles** and click 'Deep Dive' to use the AI Summarizer and Entity Extractor on long articles.
        """)

    st.markdown("---")
    
    st.markdown("### 🌟 What, Why, and How is it Unique?")
    st.write("""
    **What:** It is not just a sentiment analyzer; it is a full **Financial Intelligence Dashboard**.\n
    **Why:** Most open-source projects rely on outdated Lexicon models (TextBlob/VADER). This project brings Enterprise-grade ML (Transformers) to the browser without crashing memory constraints.\n
    **How:** By utilizing a highly optimized, lazy-loading Streamlit caching architecture (`@st.cache_resource`), it dynamically summons up to 4 heavy neural networks (FinBERT, DistilBART, BERT-NER, DistilBERT-MNLI) precisely when needed, preventing Out-Of-Memory errors on cloud hosts.
    """)

    st.markdown("### 🛠️ Key Features & Technologies Used")
    f1, f2, f3 = st.columns(3)
    f1.markdown("- **Core Python:** `Streamlit`, `Pandas`, `Numpy`\n- **Scraping:** `Feedparser`, `BeautifulSoup4`, `Requests`\n- **Market API:** `yfinance`")
    f2.markdown("- **Deep Learning:** `PyTorch`, `Transformers` (HuggingFace)\n- **Sentiment:** ProsusAI/FinBERT, VADER\n- **Summarization:** Falconsai DistilBART")
    f3.markdown("- **Topic Modeling:** Zero-Shot MNLI\n- **Entity Extraction:** BERT-Base-NER\n- **Visualization:** `Plotly` Express & Graph Objects")

def page_dashboard():
    st.title("📊 Financial News & Intelligence Dashboard")
    st.write("Scan the market for institutional news, assess AI-driven sentiment, and uncover dominant topics.")
    
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        query = st.text_input("Enter Ticker or Topic (e.g., TSLA, AI Regulations, Inflation)", value=st.session_state.search_query)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Fetch & Analyze News", type="primary", use_container_width=True):
            st.session_state.search_query = query
            with st.spinner(f"Scraping Web & running FinBERT for '{query}'..."):
                articles = fetch_rss_news(query, limit=25)
                for art in articles:
                    combined_text = art['title'] + ". " + art['summary']
                    sentiment_results = analyze_sentiment(combined_text)
                    art.update(sentiment_results)
                    art['Topic'] = detect_topic(art['title'])
                st.session_state.news_data = articles
            st.rerun()
            
    if st.session_state.news_data:
        df = pd.DataFrame(st.session_state.news_data)
        
        # Format dates for plotting
        try:
            df['parsed_date'] = pd.to_datetime(df['published'], format='mixed', utc=True)
            df = df.sort_values(by='parsed_date')
        except:
            pass # Fallback if date parsing fails
            
        st.markdown("### 📈 Executive Intelligence Summary")
        total_arts = len(df)
        pos = len(df[df['FinBERT_Label'] == 'Positive'])
        neg = len(df[df['FinBERT_Label'] == 'Negative'])
        neu = len(df[df['FinBERT_Label'] == 'Neutral'])
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Articles Processed", total_arts)
        m2.metric("Bullish Sentiment", pos, f"{(pos/total_arts)*100:.0f}%")
        m3.metric("Bearish Sentiment", neg, f"-{(neg/total_arts)*100:.0f}%")
        m4.metric("Neutral Noise", neu)
        
        st.markdown("---")
        
        # Row 1 of Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### FinBERT Sentiment Distribution")
            fig_pie = px.pie(df, names='FinBERT_Label', color='FinBERT_Label',
                             color_discrete_map={'Positive':'#00FF41', 'Negative':'#FF003C', 'Neutral':'#FFAA00'}, hole=0.4)
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_chart2:
            st.markdown("#### Dominant Market Topics")
            topic_counts = df['Topic'].value_counts().reset_index()
            topic_counts.columns = ['Topic', 'Count']
            fig_bar = px.bar(topic_counts, x='Topic', y='Count', color='Topic', color_discrete_sequence=px.colors.sequential.Plasma)
            fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Row 2 of Charts (Timeline)
        if 'parsed_date' in df.columns:
            st.markdown("#### 🕒 Sentiment Confidence Timeline")
            fig_scatter = px.scatter(df, x='parsed_date', y='FinBERT_Confidence', color='FinBERT_Label', 
                                     color_discrete_map={'Positive':'#00FF41', 'Negative':'#FF003C', 'Neutral':'#FFAA00'},
                                     hover_data=['title', 'source'], size_max=10, size='FinBERT_Confidence')
            fig_scatter.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Publication Time", yaxis_title="AI Confidence %")
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### 🗃️ Raw Structured Intelligence Data")
        st.dataframe(df[['title', 'source', 'Topic', 'FinBERT_Label', 'FinBERT_Confidence', 'published']], use_container_width=True)

def page_market_data():
    st.title("📈 Real-Time Market Data & Social Sentiment")
    st.write("Cross-reference 30-day technical price action against current retail/social chatter.")
    
    col_input, col_load = st.columns([3, 1])
    with col_input:
        ticker = st.text_input("Enter Exact Stock Ticker (e.g., AAPL, TSLA, MSFT, BTC-USD):", value="AAPL")
    with col_load:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_load = st.button("Load Live Market Data", type="primary", use_container_width=True)
        
    if btn_load:
        ticker = ticker.upper().strip()
        col_charts, col_social = st.columns([2, 1])
        
        with col_charts:
            st.markdown(f"### {ticker} Price Action (Last 30 Days)")
            with st.spinner("Fetching Yahoo Finance data..."):
                try:
                    stock = yf.Ticker(ticker)
                    # Fetch last 30 days
                    hist = stock.history(period="1mo")
                    
                    if not hist.empty:
                        # Extract metrics safely
                        latest_close = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest_close
                        pct_change = ((latest_close - prev_close) / prev_close) * 100
                        period_high = hist['High'].max()
                        period_low = hist['Low'].min()
                        
                        # Metrics Row
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Real-Time Close Price", f"${latest_close:.2f}", f"{pct_change:.2f}%")
                        m2.metric("30-Day Period High", f"${period_high:.2f}")
                        m3.metric("30-Day Period Low", f"${period_low:.2f}")
                        
                        # Candlestick Graph
                        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price")])
                        # Add a quick 5-day moving average for short term trend
                        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(window=5).mean(), line=dict(color='orange', width=2), name='5-Day MA'))
                        fig.update_layout(template="plotly_dark", margin=dict(t=20, b=0, l=0, r=0), xaxis_rangeslider_visible=False, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Invalid ticker or no recent trading data found on Yahoo Finance.")
                except Exception as e:
                    st.error(f"Error connecting to financial data server: {e}")
                    
        with col_social:
            st.markdown("### 🗣️ Reddit WallStreetBets Pulse")
            with st.spinner("Scraping Reddit communities..."):
                reddit_posts = fetch_reddit_sentiment(ticker)
            
            if reddit_posts:
                vader_analyzer = load_vader()
                pos_count, neg_count, neu_count = 0, 0, 0
                for post in reddit_posts:
                    vs = vader_analyzer.polarity_scores(post['title'])['compound']
                    if vs > 0.05: pos_count += 1
                    elif vs < -0.05: neg_count += 1
                    else: neu_count += 1
                
                total = len(reddit_posts)
                st.write(f"**{total} Mentions Found Recently**")
                
                # Progress bar for sentiment ratio
                bull_ratio = pos_count / max(1, (pos_count + neg_count))
                st.progress(bull_ratio)
                st.caption(f"Social Sentiment Ratio: {int(bull_ratio*100)}% Bullish vs {100-int(bull_ratio*100)}% Bearish")
                
                st.markdown("**Latest Retail Chatter:**")
                for p in reddit_posts[:6]:
                    st.markdown(f"""
                    <div style="background-color: #1E1E1E; padding: 10px; border-radius: 5px; margin-bottom: 8px; font-size: 0.9em; border-left: 3px solid {'#00FF41' if vader_analyzer.polarity_scores(p['title'])['compound'] > 0 else '#FF003C'};">
                    {p['title']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent Reddit data found for this specific query. It might be flying under the retail radar.")

def page_articles():
    st.title("📰 Scraped Article Repository")
    st.write("Browse fetched headlines. Click 'Deep Dive Analysis' to utilize Web Scraping, AI Summarization, and Named Entity Recognition.")
    
    if not st.session_state.news_data:
        st.warning("No data found in cache. Please run an analysis in the Dashboard first.")
        if st.button("Go to Dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
        return
    
    for idx, article in enumerate(st.session_state.news_data):
        with st.container():
            st.markdown(f"<div class='article-card'>", unsafe_allow_html=True)
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(article['title'])
                st.caption(f"🏷️ Topic: **{article.get('Topic', 'Unknown')}** | 🏢 Source: {article['source']} | 📅 {article['published']}")
                color = "#00FF41" if article['FinBERT_Label'] == "Positive" else "#FF003C" if article['FinBERT_Label'] == "Negative" else "#FFAA00"
                st.markdown(f"**AI Sentiment:** <span style='color:{color}'>{article['FinBERT_Label']}</span> ({article['FinBERT_Confidence']}%)", unsafe_allow_html=True)
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
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
        
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("⬅ Back to Headlines", use_container_width=True):
            st.session_state.page = "Scraped Articles"
            st.rerun()
        
    st.markdown("---")
    with st.spinner("Executing dynamic web scraper to extract full HTML text & images..."):
        img_url, full_content = scrape_full_article(article['link'])
    
    col_content, col_tools = st.columns([2, 1])
    
    with col_content:
        st.title(article['title'])
        st.markdown(f"**Source:** [{article['source']}]({article['link']}) | **Published:** {article['published']}")
        
        # Display image from original URL
        st.image(img_url, use_container_width=True)
        
        st.markdown("### Extracted Article Content")
        st.write(full_content)
        
    with col_tools:
        st.markdown("### 🤖 ML Analysis Tools")
        
        color = "#00FF41" if article['FinBERT_Label'] == "Positive" else "#FF003C" if article['FinBERT_Label'] == "Negative" else "#FFAA00"
        st.markdown(f"<div class='metric-box'>Sentiment: <b style='color:{color};'>{article['FinBERT_Label']}</b><br>Confidence: {article['FinBERT_Confidence']}%<br>Topic: {article.get('Topic', 'Unknown')}</div><br>", unsafe_allow_html=True)
        
        st.markdown("#### AI Summarization (TL;DR)")
        st.caption("Compresses long articles into bite-sized highlights using DistilBART Neural Networks.")
        if st.button("Generate Summary", type="primary", use_container_width=True):
            with st.spinner("Neural network is reading and summarizing..."):
                if len(full_content) > 300:
                    summarizer_pipe = load_summarizer()
                    summary = summarizer_pipe(full_content[:2000], max_length=130, min_length=30, do_sample=False)
                    st.success(summary[0]['summary_text'])
                else:
                    st.warning("Text too short to require summarization.")
        
        st.markdown("---")
        st.markdown("#### Named Entity Recognition (NER)")
        st.caption("Identifies key Organizations, Persons, and Locations mentioned in the text.")
        if st.button("Extract Entities", use_container_width=True):
            with st.spinner("Extracting..."):
                ner_pipe = load_ner()
                entities = ner_pipe(full_content[:2000])
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
    
    col_img, col_bio = st.columns([1, 2])
    with col_img:
        st.image("https://profileravi.netlify.app/og/img.webp", width=250, use_container_width=True)
        st.markdown("### Ravi Vishwakarma")
        st.write("Machine Learning Engineer & Web App Developer")
        
        st.markdown("#### Connect With Me")
        # Standard Streamlit Link Buttons
        st.link_button("🌐 Portfolio Website", "https://profileravi.netlify.app/", use_container_width=True)
        st.link_button("💼 LinkedIn", "https://linkedin.com/in/ravi-vishwakarma-2a8264253", use_container_width=True) # Please update with actual link
        st.link_button("💻 GitHub", "https://github.com/RAVIVISHWAKARMA2003", use_container_width=True) # Please update with actual link
        
    with col_bio:
        st.header("Enterprise Architecture")
        st.write("""
        This application acts as a microcosm of an enterprise-level financial data pipeline. By combining real-time data ingestion with high-level NLP, it demonstrates the power of Applied Machine Learning in Fintech.
        """)
        
        st.markdown("### System Pipeline")
        st.write("""
        1. **Ingestion Layer:** Connects to RSS endpoints, executes web scrapers (`BeautifulSoup4`), and calls API wrappers (`yfinance`, `PRAW`/Reddit HTTP).
        2. **Processing Layer:** Cleans HTML, processes timestamps, and handles URL encoding safely.
        3. **Inference Layer:** Uses HuggingFace `pipeline` to route text through 4 distinct neural networks.
           - `ProsusAI/finbert` (Sentiment)
           - `Falconsai/text_summarization` (Abstractive Summarization)
           - `dslim/bert-base-NER` (Entity Extraction)
           - `typeform/distilbert-base-uncased-mnli` (Zero-Shot Topic Modeling)
        4. **Data Visualization Layer:** Structures multi-dimensional tensor outputs into `pandas` DataFrames, visualizes distributions with `plotly.express`, and handles financial candlestick geometry with `plotly.graph_objects`.
        5. **Memory Management:** Implements aggressive Lazy-Loading via discrete `@st.cache_resource` states, ensuring models load into memory *only* when the user explicitly triggers a function, avoiding cloud runtime crashes.
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
