import torch
from ml_loader import get_finbert, get_vader, get_topic_model

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
