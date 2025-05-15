# Restaurant Review Analysis & Chatbot System

This repository contains a pipeline for analyzing restaurant review data, including:

* Text preprocessing and exploration
* Fine-tuned PhoBERT-based models for:

  * Sentiment analysis (NEG, NEU, POS)
  * Multi-label classification (food, service, delivery, price, ambience)
* A chatbot backend framework using a Vietnamese language model for food-related Q\&A

## Project Structure

```
.
├── data-processing.ipynb        # Notebook for data cleaning and preparation
├── model_training.ipynb         # Notebook for fine-tuning PhoBERT models
├── pipeline.py                  # Main pipeline script for sentiment and classification
├── chatbot/                     # Chatbot backend service
│   ├── app.py                   # FastAPI app with ngrok integration
│   ├── config/
│   │   ├── constant.py
│   │   ├── model.py
│   │   └── prompt.py
│   ├── module/
│   │   ├── graph/
│   │   │   ├── agent.py
│   │   │   ├── nodes.py
│   │   │   └── tools.py
│   │   ├── rabbitmq/
│   │   │   ├── message_handler.py
│   │   │   └── message_listener.py
│   │   ├── storage/
│   │   │   ├── blob.py
│   │   │   └── checkpointer.py
│   │   └── tool/
│   │       ├── code_interpreter.py
│   │       ├── helper.py
│   │       └── chat_model.py
│   └── utils/
│       └── logger.py
└── .gitignore
```

## Review Analysis Pipeline

* Segment each review into components
* Predict sentiment per component
* Assign categories based on content
* Compute adjusted scores based on sentiment balance

### Run

```bash
python pipeline.py
```

## Chatbot API

* Built with FastAPI
* Uses a Vietnamese language model from Hugging Face for Q\&A
* Accepts optional image uploads

### Run Server

```bash
cd chatbot
python app.py
```

Access the API documentation at: `[chatbot](https://precious-needed-bug.ngrok-free.app/docs#/Chat/chat_chat_post)`

## Requirements

```bash
pip install torch pandas transformers tqdm fastapi uvicorn ngrok rich python-dotenv pillow opencv-python
```
## Authors

Team 7 – Grab Tech Bootcamp 2025
