import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import health, analysis, article, market, websocket as ws_router

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

@app.on_event("startup")
def startup_event():
    import threading
    def load_all_models():
        print("Pre-loading Machine Learning models in background...")
        try:
            from ml_loader import get_vader, get_finbert, get_topic_model, get_summarizer, get_ner
            # Load Vader
            get_vader()
            # Load FinBERT
            get_finbert()
            # Load Topic Model
            get_topic_model()
            # Load Summarizer
            get_summarizer()
            # Load NER
            get_ner()
            print("Pre-loading finished successfully! All models loaded in RAM.")
        except Exception as e:
            print(f"Error pre-loading models: {e}")
            
    threading.Thread(target=load_all_models, daemon=True).start()

# Register routers
app.include_router(health.router)
app.include_router(analysis.router)
app.include_router(article.router)
app.include_router(market.router)
app.include_router(ws_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
