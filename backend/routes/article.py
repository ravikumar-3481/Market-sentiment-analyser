from fastapi import APIRouter, HTTPException
from typing import List
from schemas import ScrapeRequest, ScrapeResponse, ContentRequest, SummarizeResponse, EntityResponse
from news_scraper import scrape_full_article
from ml_loader import get_summarizer, get_ner

router = APIRouter()

@router.post("/api/scrape-article", response_model=ScrapeResponse)
def scrape_article(request: ScrapeRequest):
    try:
        image_url, content = scrape_full_article(request.url)
        return {"image_url": image_url, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/summarize", response_model=SummarizeResponse)
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

@router.post("/api/extract-entities", response_model=EntityResponse)
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
