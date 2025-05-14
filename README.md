# Sentiment Analysis & Classification Pipeline

## Overview

This repository contains scripts and notebooks for processing and analyzing restaurant review data. It utilizes transformer-based models, specifically fine-tuned PhoBERT, to perform sentiment analysis and multi-label classification on Vietnamese textual reviews.

## File Descriptions

### Notebooks

* `data-processing.ipynb`: Contains steps for preprocessing, cleaning, and preparing raw review data for model training and evaluation.
* `model_training.ipynb`: Includes training procedures, evaluation metrics, and visualization of results for sentiment analysis and review classification models.

### Scripts

* `pipeline.py`: Implements the entire sentiment and classification pipeline, including sentence segmentation, sentiment prediction, and multi-label classification. The script processes input CSV files containing restaurant reviews and outputs predictions in CSV format.

## Models Used

* **Sentiment Analysis:** PhoBERT fine-tuned for sentiment classification (`checkpoint-1100-sentiment`).
* **Multi-label Classification:** PhoBERT fine-tuned for categorizing reviews into `ambience`, `delivery`, `food`, `price`, and `service` (`checkpoint-510-classification`).

## Usage

Run the pipeline script:

```bash
python pipeline.py
```

Modify input and output CSV paths directly in `pipeline.py`:

```python
input_csv = "path/to/input_reviews.csv"
output_csv = "path/to/output_predictions.csv"
```

## Requirements

* Python 3.xx
* pandas
* torch
* transformers
* tqdm

Install dependencies:

```bash
pip install pandas torch transformers tqdm
```

## Contributions

This project is developed for the Grab Tech Bootcamp 2025, Team 7. Contributions and improvements are welcome.

---

Team 7 Grab Tech Bootcamp 2025
