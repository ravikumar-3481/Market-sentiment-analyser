# 🚀 Deploying FastAPI Backend to Render

This guide outlines how to deploy the **MarketPulse AI** FastAPI backend on **Render**.

---

## 🛠️ Option 1: Manual Dashboard Setup (Recommended)

When creating a new **Web Service** on the [Render Dashboard](https://dashboard.render.com/):

1. **Repository**: Connect your GitHub/GitLab repository.
2. **Name**: `marketpulse-ai-backend` (or your preferred name)
3. **Environment/Runtime**: `Python`
4. **Root Directory**: `backend` *(Crucial: This ensures Render runs commands relative to the `backend` folder)*
5. **Build Command**: 
   ```bash
   chmod +x build.sh && ./build.sh
   ```
   *(Or simply: `pip install -r requirements.txt`)*
6. **Start Command**: 
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

---

## 📄 Option 2: Render Blueprint Setup (Automated)

A [render.yaml](file:///f:/code/project/Market-sentiment-analyser/render.yaml) file has been created at the root of the repository.

To deploy using the Blueprint:
1. Go to **Blueprints** on the Render Dashboard.
2. Click **New Blueprint Instance**.
3. Select your repository. Render will automatically read the `render.yaml` configuration and set up the service.

---

## ⚠️ Important Deployment Notes for Free Tier Users

> [!WARNING]
> The backend imports **PyTorch (`torch`)** and **HuggingFace Transformers (`transformers`)** for FinBERT, Zero-Shot Topic Classification, and DistilBART.
> - **Memory Limits**: Render's Free Tier provides **512MB RAM**. Loading multiple HuggingFace models in RAM will likely cause the container to crash with an **Out Of Memory (OOM)** error.
> - **Mitigation Options**:
>   1. **Upgrade Instance**: Upgrade to a starter or standard instance on Render (at least 1GB to 2GB RAM).
>   2. **CPU-only Torch**: You can edit [requirements.txt](file:///f:/code/project/Market-sentiment-analyser/backend/requirements.txt) to install a CPU-only PyTorch build to reduce the container's disk space and memory footprint:
>      ```
>      --extra-index-url https://download.pytorch.org/whl/cpu
>      torch>=2.0.0
>      ```
