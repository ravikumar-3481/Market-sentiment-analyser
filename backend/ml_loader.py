# Lazy loading components for machine learning models to save RAM/load time.

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
