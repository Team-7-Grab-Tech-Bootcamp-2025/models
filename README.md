# Restaurant Review Analysis & Chatbot System

This repository contains a full pipeline for analyzing restaurant review data, including:

* Text preprocessing and exploration
* Fine-tuned PhoBERT-based models for:

  * Sentiment analysis (NEG, NEU, POS)
  * Multi-label classification (food, service, delivery, price, ambience)
* A chatbot backend framework for interactive Q\&A on food-related topics

## Project Structure

```
.
├── data-processing.ipynb        # Data cleaning and preparation notebook
├── model_training.ipynb         # Notebook for fine-tuning PhoBERT models
├── pipeline.py                  # Main pipeline script for classification
├── chatbot/                     # Chatbot backend service
│   ├── app.py                   # FastAPI app with ngrok integration
│   ├── config/                  # Configurations and constants
│   │   ├── constant.py          # Environment and system config
│   │   ├── model.py             # Pydantic schemas for chat model
│   │   └── prompt.py            # Prompts used by the chatbot
│   ├── tool/
│   │   └── chat_model.py        # ChatClient implementation with ViQwen model
│   └── utils/
│       └── logger.py            # Custom logger with timestamp formatting
└── .gitignore
```

## Review Analysis Pipeline

* Segment each review into components
* Predict sentiment per component
* Assign categories based on content
* Compute re-rated scores based on sentiment mix

### Run

```bash
python pipeline.py
```

## Chatbot API

* Built on FastAPI
* Integrates ViQwen 1.5B model for dialogue generation
* Supports optional image input

### Run server

```bash
cd chatbot
python app.py
```

Visit: [http://localhost:31456/docs](http://localhost:31456/docs) for Swagger UI

## Environment Variables

```env
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
MONGODB_URI=...
SYSTEM_OS_PATH=...
```

## Requirements

```bash
pip install torch pandas transformers tqdm fastapi uvicorn ngrok rich python-dotenv pillow opencv-python
```

## Model Checkpoints

| Task                       | Model Name                              |
| -------------------------- | --------------------------------------- |
| Sentiment Analysis         | checkpoint-1100-sentiment (PhoBERT)     |
| Multi-label Classification | checkpoint-510-classification (PhoBERT) |
| Chatbot Dialogue           | AITeamVN/Vi-Qwen2-1.5B-RAG              |

## Authors

Team 7 – Grab Tech Bootcamp 2025

---

© 2025 Team 7 | All rights reserved.
