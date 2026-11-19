# KidneyVision

KidneyVision is an integrated medical AI web application that pairs deep learning CT scan analysis with an intelligent patient-education chatbot.

## Features

- **CT Scan Classification & Segmentation**: Powered by a trained Swin Transformer model (`models/weights/model.pth`), detecting and segmenting stones, cysts, and tumors.
- **Segmentation PDF Export**: Generates a one-page downloadable medical report with highlighted CT and segmentation mask overlays.
- **Kidney Assistant Chatbot**: RAG-based educational chatbot backed by clinical knowledge base topics (`data/raw/kidney_knowledge.docx`) with multi-provider LLM failover resilience (Gemini, Groq, Mistral, ZAI, Cloudflare).
- **Direct Integrated Web Frontend**: Directly served by Flask with real-time UI interaction, no separate API proxy or extra web server required.

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- Dependencies installed:
  ```bash
  pip install -r requirements.txt
  ```

### Configuration

Copy `.env.example` to `.env` and set your API keys as desired:

```bash
cp .env.example .env
```

### Running the Application

To launch both the backend and frontend together:

```bash
python app.py
```

The application starts on `http://127.0.0.1:5000/` and will automatically open in your default browser.

## Running Tests

Run the full automated test suite:

```bash
python -m unittest test_app.py check_flask_backend.py check_provider_adapters.py check_llm_fallback.py
```

## Project Structure

```text
backend/
├── app.py                      # Flask web application & API entrypoint
├── chatbot.py                  # RAG chatbot with multi-provider fallback
├── knowledge_base.py           # Document parser & semantic retrieval
├── llm_providers.py            # Unified LLM provider adapters
├── provider_resilience.py      # Circuit breaker & retry strategies
├── model_predictor.py          # Swin model inference & segmentation pipeline
├── segmentation_pdf.py         # ReportLab PDF generator
├── requirements.txt            # Runtime dependencies
├── test_app.py                 # Integrated application tests
├── check_flask_backend.py      # Conversation isolation & routing tests
├── check_provider_adapters.py  # LLM adapter tests
├── check_llm_fallback.py       # LLM fallback resilience tests
├── data/                       # Knowledge base source documents
├── frontend/                   # Web frontend assets (HTML, CSS, JS, images)
├── inference_v2/               # Model inference architecture & modules
└── models/weights/             # Model checkpoint weights (model.pth)
```
