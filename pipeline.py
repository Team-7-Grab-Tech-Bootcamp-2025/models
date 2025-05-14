import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from tqdm import tqdm
import uuid

# 1. Load models
sentiment_model_id = r"C:\Users\vumin\Desktop\Grab\checkpoint-1100-sentiment"
classify_model_path = r"C:\Users\vumin\Desktop\Grab\checkpoint-510-classification"  # Thư mục chứa model đã fine-tuned với 5 nhãn

sentiment_tokenizer = AutoTokenizer.from_pretrained(
    "wonrax/phobert-base-vietnamese-sentiment", use_fast=False
)
# sentiment_tokenizer.save_pretrained(sentiment_model_id)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_id)
sentiment_model.eval()

classify_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
# classify_tokenizer.save_pretrained(sentiment_model_id)
classify_model = AutoModelForSequenceClassification.from_pretrained(
    classify_model_path, num_labels=5, problem_type="multi_label_classification"
)
classify_model.eval()

label_names = ["ambience", "delivery", "food", "price", "service"]
sentiment_map = {0: "NEG", 1: "NEU", 2: "POS"}

# 2. Hàm tách câu (component)
conjs = [
    "nhưng",
    "mà",
    "bởi vì",
    "tuy nhiên",
    "và",
    "hay",
    "hoặc",
    "hoặc là",
    "cũng như",
    "vì vậy",
    "do đó",
    "thế nhưng",
    "thế mà",
    "dù sao đi nữa",
    "cũng như là",
    "lẫn",
    "cùng",
    "ngoài ra",
    "vì",
    "bởi vì",
    "nên",
    "vậy",
    "do đó",
    "nếu",
    "trừ phi",
    "trừ khi",
    "hễ",
    "hơn",
    "bằng",
    "như",
    "mặc dù",
    "dù chơ",
    "tuy nhiên",
    "sau khi",
    "khi",
    "để nhằm",
    "mục đích",
    "ngược lại",
]
pattern = r"(?:[\.!\?;,]+|\b(?:" + "|".join(map(re.escape, conjs)) + r")\b)\s*"


def segment_sentences(text):
    parts = re.split(pattern, str(text))
    return [p.strip() for p in parts if p.strip()]


# 3. Hàm dự đoán cảm xúc
def predict_sentiment(texts):
    results = []
    for text in texts:
        inputs = sentiment_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs).item()
            results.append(sentiment_map[pred])
    return results


# 4. Hàm phân loại component
def classify_components(texts):
    texts = [t for t in texts if t.strip()]
    if not texts:
        return [[]]

    try:
        inputs = classify_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = classify_model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).int()

        return [
            [label_names[i] for i, v in enumerate(row) if v == 1]
            for row in preds.cpu().numpy()
        ]

    except Exception as e:
        print(f"❗ Bỏ qua vì lỗi khi classify: {e}")
        return [[]] * len(texts)


# 5. Pipeline chính
def run_pipeline(input_csv, output_csv):
    df = pd.read_csv(input_csv, quoting=1, encoding="utf-8", engine="python")

    feedback_id = 9207

    # Ghi header trước
    columns = [
        "feedback_id",
        "rating_id",
        "component",
        "sentiment",
        "category",
        "re_rating",
    ]
    pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        rating_id = row.get("rating_id")
        original_rating = row.get("rating", 5)

        # ✅ Trường hợp không có review_text hoặc là "nan"
        review_text = str(row.get("review_text", "")).strip()
        if not review_text or review_text.lower() == "nan":
            row_data = {
                "feedback_id": feedback_id,
                "rating_id": rating_id,
                "component": [""],
                "sentiment": "NEU",
                "category": "unknown",
                "re_rating": original_rating,
            }
            pd.DataFrame([row_data]).to_csv(
                output_csv, mode="a", index=False, header=False
            )
            feedback_id += 1
            continue

        components = segment_sentences(review_text)
        sentiments = predict_sentiment(components)
        categories = classify_components(components)

        # Gom theo (category, sentiment)
        group_map = {}
        all_sentiments = set()
        for comp, sent, cats in zip(components, sentiments, categories):
            all_sentiments.add(sent)
            for cat in cats:
                key = (cat, sent)
                group_map.setdefault(key, []).append(comp)

        # Đếm cảm xúc
        sent_counts = {"POS": 0, "NEG": 0, "NEU": 0}
        for s in sentiments:
            sent_counts[s] += 1

        n_pos, n_neg, n_neu = sent_counts["POS"], sent_counts["NEG"], sent_counts["NEU"]
        total = n_pos + n_neg + n_neu
        unique_sentiments = [s for s, c in sent_counts.items() if c > 0]

        # Tính rating_map
        if len(unique_sentiments) == 1:
            if n_pos > 0:
                rating_map = {"POS": 5, "NEG": 5, "NEU": 5}
            elif n_neg > 0:
                rating_map = {"POS": 1, "NEG": 1, "NEU": 1}
            else:
                rating_map = {s: original_rating for s in sent_counts}
        else:
            x = (n_neg - n_pos) / (n_pos + n_neg) if (n_neg - n_pos) > 0 else 0
            rating_map = {
                "POS": round(original_rating + x, 2),
                "NEG": round(original_rating - x, 2),
                "NEU": original_rating,
            }

        # Ghi từng dòng ra file
        for (cat, sent), comps in group_map.items():
            re_rating = max(1, min(5, rating_map.get(sent, original_rating)))
            row_data = {
                "feedback_id": feedback_id,
                "rating_id": rating_id,
                "component": comps,
                "sentiment": sent,
                "category": cat,
                "re_rating": re_rating,
            }
            pd.DataFrame([row_data]).to_csv(
                output_csv, mode="a", index=False, header=False
            )
            feedback_id += 1

    print(f"✅ Output saved to {output_csv}")


if __name__ == "__main__":
    input_csv = r"C:\Users\vumin\Desktop\Grab\data\data_grb\foody\foody_reviews.csv"  # Đường dẫn đến file CSV đầu vào
    output_csv = "foody_predicted_sentiment_2.csv"  # Đường dẫn đến file CSV đầu ra
    run_pipeline(input_csv, output_csv)
