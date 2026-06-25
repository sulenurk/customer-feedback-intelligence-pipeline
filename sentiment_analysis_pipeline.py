import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import random

# ==========================================
# 1. SETTINGS
# ==========================================
EXCEL_FILE = "review.xlsx"
COLUMN_NAME = "Review"
OUTPUT_DIR = "./outputs"
MODEL_PATH = "./checkpoint"

BRAND_ORANGE = "#FCA311"
BRAND_NAVY = "#14213D"
BRAND_BG = "#F8FAFC"
BRAND_LIGHT = "#E5E7EB"
BRAND_MUTED = "#64748B"

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": BRAND_BG,
    "axes.facecolor": BRAND_BG,
    "axes.edgecolor": BRAND_LIGHT,
    "axes.labelcolor": BRAND_NAVY,
    "xtick.color": BRAND_NAVY,
    "ytick.color": BRAND_NAVY,
    "text.color": BRAND_NAVY,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold"
})

print("Reading dataset...")
df = pd.read_excel(EXCEL_FILE)
df_sample = df.copy()

# ==========================================
# 2. LOAD MODELS (Sentiment + Summarization)
# ==========================================
print("Loading sentiment model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

print("Loading summarization model (BART)...")
# Background model download and optimization via pipeline.
# Local caching is handled automatically by Hugging Face.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

import torch.nn.functional as F

def predict_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral"

    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128, 
        return_token_type_ids=False
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # 1. Ham çıktıları (logits) gerçek yüzdelik olasılıklara (probabilities) çeviriyoruz
        probs = F.softmax(outputs.logits, dim=-1)[0]
        
    neg_prob = probs[0].item()  # 0: Negative olasılığı
    pos_prob = probs[1].item()  # 1: Positive olasılığı

    # 2. THRESHOLD LOGIC
# Labels as "Neutral" if confidence is < 75% to avoid data contamination.
    CONFIDENCE_THRESHOLD = 0.75

    if neg_prob > CONFIDENCE_THRESHOLD:
        return "Negative"
    elif pos_prob > CONFIDENCE_THRESHOLD:
        return "Positive"
    else:
        return "Neutral"

# ==========================================
# 3. SENTIMENT ANALYSIS
# ==========================================
print("Running sentiment analysis...")
df_sample["Sentiment"] = df_sample[COLUMN_NAME].apply(predict_sentiment)

# ==========================================
# Advanced Filter: Fixing Labels Before Plotting
# ==========================================
positive_triggers = ["exceeds expectations", "very slowly", "simple", "impressed", "great", "excellent", "love"]

def fix_slipped_sentiments(row):
    review_text = str(row[COLUMN_NAME]).lower()
    # Change label if model predicts negative despite obvious positive patterns
    if row["Sentiment"] == "Negative":
        if any(trigger in review_text for trigger in positive_triggers):
            return "Neutral" 
    return row["Sentiment"]

# Run this line before visualization/plotting:
df_sample["Sentiment"] = df_sample.apply(fix_slipped_sentiments, axis=1)

# ==========================================
# 4. MODERN SENTIMENT DISTRIBUTION CHART
# ==========================================
print("Creating sentiment distribution chart...")

counts = df_sample["Sentiment"].value_counts().reindex(["Positive", "Negative"]).fillna(0)
total = counts.sum()
percentages = (counts / total * 100) if total > 0 else counts

fig, ax = plt.subplots(figsize=(8, 4.8))
fig.patch.set_facecolor(BRAND_BG)
ax.set_facecolor(BRAND_BG)

labels = counts.index.tolist()
values = counts.values
colors = [BRAND_ORANGE, BRAND_NAVY]

bars = ax.barh(
    labels,
    values,
    color=colors,
    height=0.42,
    edgecolor="none"
)

ax.set_title(
    "Sentiment Distribution",
    fontsize=15,
    fontweight="semibold",
    color=BRAND_NAVY,
    pad=34,
    loc="left"
)

ax.text(
    0,
    1.08,
    "Overview of predicted sentiment labels in the review sample",
    transform=ax.transAxes,
    fontsize=10.5,
    color=BRAND_MUTED,
    ha="left"
)

ax.set_xlabel("Number of Reviews", fontsize=10, color=BRAND_MUTED, labelpad=12)
ax.set_ylabel("")

ax.grid(axis="x", linestyle="-", linewidth=0.8, alpha=0.18)
ax.set_axisbelow(True)

for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="y", labelsize=11, length=0, pad=10)
ax.tick_params(axis="x", labelsize=9, colors=BRAND_MUTED)

max_value = max(values) if len(values) > 0 and max(values) > 0 else 1

for bar, value, pct in zip(bars, values, percentages.values):
    ax.text(
        value + max_value * 0.03,
        bar.get_y() + bar.get_height() / 2,
        f"{int(value)} reviews  ·  {pct:.1f}%",
        va="center",
        ha="left",
        fontsize=10.5,
        fontweight="medium",
        color=BRAND_NAVY
    )

ax.set_xlim(0, max_value * 1.35)

plt.tight_layout(pad=2.4)

histogram_path = os.path.join(OUTPUT_DIR, "sentiment_distribution_5.png")
plt.savefig(histogram_path, dpi=300, bbox_inches="tight", facecolor=BRAND_BG)
plt.close()

print(f"Saved chart: {histogram_path}")


# ==========================================
# 5. MODERN WORD CLOUDS
# ==========================================
print("Creating word clouds...")

custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    "br", "will", "one", "also", "really", "much", "even"
])

positive_text = " ".join(
    df_sample[df_sample["Sentiment"] == "Positive"][COLUMN_NAME]
    .dropna()
    .astype(str)
)

negative_text = " ".join(
    df_sample[df_sample["Sentiment"] == "Negative"][COLUMN_NAME]
    .dropna()
    .astype(str)
)

def positive_color_func(*args, **kwargs):
    return random.choice([BRAND_ORANGE, "#FFB703", "#FB8500", BRAND_NAVY])

def negative_color_func(*args, **kwargs):
    return random.choice([BRAND_NAVY, "#334155", "#475569", BRAND_ORANGE])

def create_wordcloud(text, title, subtitle, filename, color_func):
    if len(text.strip()) <= 5:
        print(f"Not enough text to create {title}.")
        return

    wordcloud = WordCloud(
        width=1400,
        height=900,
        background_color=BRAND_BG,
        stopwords=custom_stopwords,
        max_words=120,
        prefer_horizontal=0.92,
        collocations=False,
        random_state=42,
        margin=8
    ).generate(text)

    wordcloud.recolor(color_func=color_func, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.patch.set_facecolor(BRAND_BG)
    ax.set_facecolor(BRAND_BG)

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    fig.text(
        0.08,
        0.94,
        title,
        fontsize=15,
        fontweight="semibold",
        color=BRAND_NAVY,
        ha="left"
    )

    fig.text(
        0.08,
        0.895,
        subtitle,
        fontsize=10.5,
        color=BRAND_MUTED,
        ha="left"
    )

    plt.subplots_adjust(top=0.78, left=0.04, right=0.96, bottom=0.04)

    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BRAND_BG)
    plt.close()

    print(f"Saved word cloud: {output_path}")

create_wordcloud(
    positive_text,
    "Positive Review Keywords",
    "Most frequent terms appearing in reviews predicted as positive",
    "positive_wordcloud_5.png",
    positive_color_func
)

create_wordcloud(
    negative_text,
    "Negative Review Keywords",
    "Most frequent terms appearing in reviews predicted as negative",
    "negative_wordcloud_5.png",
    negative_color_func
)

# ==========================================
# 6. TEXT SUMMARIZATION (NEW MODULE) - UPDATED
# ==========================================
print("Generating executive summaries...")

# 🎯 ANTI-LEAKAGE FILTER (NEW)
# Collects only definitive 'Positive' and 'Negative' reviews that clear the 75% confidence threshold.
# Prevents any 'Neutral' reviews from leaking into the final dataset.
positive_text = " ".join(df_sample[df_sample["Sentiment"] == "Positive"][COLUMN_NAME].astype(str).tolist())
negative_text = " ".join(df_sample[df_sample["Sentiment"] == "Negative"][COLUMN_NAME].astype(str).tolist())


def generate_executive_summary(text, sentiment_label):
    # Skip processing if there are no meaningful phrases for summarization
    if len(text.strip().split()) < 15:
        return f"Not enough {sentiment_label.lower()} reviews to generate a meaningful summary."
    
    # Truncating to stay within BART's 1024 input token limit
# Sends roughly the first 800 words as one chunk for summarization
    truncated_text = " ".join(text.split()[:800])
    
    try:
        summary_outputs = summarizer(
            truncated_text, 
            max_length=120, 
            min_length=35, 
            do_sample=False
        )
        return summary_outputs[0]['summary_text']
    except Exception as e:
        return f"Could not generate summary due to an error: {str(e)}"

pos_summary = generate_executive_summary(positive_text, "Positive")
neg_summary = generate_executive_summary(negative_text, "Negative")

report_path = os.path.join(OUTPUT_DIR, "executive_summary_report_5.txt")

report_content = f"""========================================================================
EXECUTIVE SUMMARY REPORT (AUTOMATED NLP INSIGHTS)
========================================================================
Generated Report Output

------------------------------------------------------------------------
🟢 POSITIVE FEEDBACK SUMMARY (What customers love):
------------------------------------------------------------------------
{pos_summary}

------------------------------------------------------------------------
🔴 NEGATIVE FEEDBACK SUMMARY (Pain points & areas of improvement):
------------------------------------------------------------------------
{neg_summary}

========================================================================
End of Report
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"Saved executive summary report: {report_path}")
print("\nAll visualizations and text reports are ready. Check the outputs folder.")

test_review = "Customer service exceeded expectations."
print(f"Test Sonucu: {predict_sentiment(test_review)}")

