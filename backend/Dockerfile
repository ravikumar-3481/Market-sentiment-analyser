# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user \
    HF_HUB_DISABLE_SYMLINKS=1

# Create a non-root user with UID 1000 to comply with Hugging Face security requirements
RUN useradd -m -u 1000 user

# Set up the working directory inside the home folder of the non-root user
WORKDIR $HOME/app

# Set PATH to include user's local bin
ENV PATH=$HOME/.local/bin:$PATH

# Copy requirements file first to leverage Docker cache
COPY --chown=user:user requirements.txt .

# Switch to the non-root user
USER user

# Upgrade pip and install CPU-only PyTorch first
# This prevents downloading the massive GPU/CUDA wheels, reducing image size and saving memory/disk space.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Install the rest of the requirements (pip will skip torch as it's already installed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY --chown=user:user . .

# Pre-download ML models at build time so they are cached inside the Docker image.
# This prevents downloading models during container startup, resulting in faster startup and preventing timeout errors.
RUN python -c " \
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline; \
print('Pre-downloading FinBERT Sentiment Engine...'); \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
print('Pre-downloading Zero-Shot Topic Classifier...'); \
pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli'); \
print('Pre-downloading AI Summarizer...'); \
pipeline('summarization', model='Falconsai/text_summarization'); \
print('Pre-downloading Entity Extractor (NER)...'); \
pipeline('ner', model='dslim/bert-base-NER'); \
print('All models pre-cached successfully!'); \
"

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Start the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
