from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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

class ScrapeRequest(BaseModel):
    url: str

class ScrapeResponse(BaseModel):
    image_url: str
    content: str

class ContentRequest(BaseModel):
    content: str

class SummarizeResponse(BaseModel):
    summary: str

class EntityResponse(BaseModel):
    ORG: List[str]
    PER: List[str]
    LOC: List[str]
    MISC: List[str]

class MarketDataResponse(BaseModel):
    ticker: str
    metrics: Dict[str, Any]
    historical_data: List[Dict[str, Any]]
    reddit_sentiment: Dict[str, Any]
