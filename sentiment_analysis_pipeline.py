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
CSV_FILE = "src/Womens Clothing E-Commerce Reviews.csv"
COLUMN_NAME = "Review Text"
OUTPUT_DIR = "./outputs2"
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
#df = pd.read_excel(EXCEL_FILE)
df = pd.read_csv(CSV_FILE)
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

def predict_sentiment_with_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series(["Neutral", 0.0, 0.0, 0.0])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_token_type_ids=False
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    neg_prob = probs[0].item()
    pos_prob = probs[1].item()
    max_prob = max(neg_prob, pos_prob)

    CONFIDENCE_THRESHOLD = 0.75

    if neg_prob > CONFIDENCE_THRESHOLD:
        label = "Negative"
    elif pos_prob > CONFIDENCE_THRESHOLD:
        label = "Positive"
    else:
        label = "Neutral"

    return pd.Series([label, neg_prob, pos_prob, max_prob])
# ==========================================
# 3. SENTIMENT ANALYSIS
# ==========================================
print("Running sentiment analysis...")
df_sample[["Sentiment", "Negative_Prob", "Positive_Prob", "Confidence"]] = (
    df_sample[COLUMN_NAME].apply(predict_sentiment_with_score)
)

# ==========================================
# Advanced Filter: Fixing Labels Before Plotting
# ==========================================
positive_triggers = [
    "exceeds expectations",
    "impressed",
    "great",
    "excellent",
    "love",
    "perfect",
    "beautiful",
    "comfortable",
    "flattering"
]

def downgrade_suspicious_negative_reviews(row):
    review_text = str(row[COLUMN_NAME]).lower()
    # Change label if model predicts negative despite obvious positive patterns
    if row["Sentiment"] == "Negative":
        if any(trigger in review_text for trigger in positive_triggers):
            return "Neutral" 
    return row["Sentiment"]

negative_triggers = [
    "not flattering",
    "runs small",
    "too small",
    "too big",
    "terrible",
    "problem",
    "stuck",
    "poor quality",
    "cheap",
    "disappointed",
    "uncomfortable",
    "bad quality",
    "had to return",
    "returning",
    "does not fit",
    "did not fit",
    "unflattering"
]

def downgrade_leaky_positive_reviews(row):
    review_text = str(row[COLUMN_NAME]).lower()

    if row["Sentiment"] == "Positive":
        if any(trigger in review_text for trigger in negative_triggers):
            return "Neutral"

    return row["Sentiment"]

df_sample["Sentiment"] = df_sample.apply(downgrade_leaky_positive_reviews, axis=1)
df_sample["Sentiment"] = df_sample.apply(downgrade_suspicious_negative_reviews, axis=1)

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

positive_reviews_for_summary = df_sample[
    (df_sample["Sentiment"] == "Positive") &
    (df_sample["Confidence"] >= 0.85)
][COLUMN_NAME].dropna().astype(str).tolist()

negative_reviews_for_summary = df_sample[
    (df_sample["Sentiment"] == "Negative") &
    (df_sample["Confidence"] >= 0.85)
][COLUMN_NAME].dropna().astype(str).tolist()

positive_text = " ".join(positive_reviews_for_summary[:50])
negative_text = " ".join(negative_reviews_for_summary[:50])

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

positive_text_wordcloud = " ".join(
    df_sample[df_sample["Sentiment"] == "Positive"][COLUMN_NAME]
    .dropna()
    .astype(str)
)

negative_text_wordcloud = " ".join(
    df_sample[df_sample["Sentiment"] == "Negative"][COLUMN_NAME]
    .dropna()
    .astype(str)
)

create_wordcloud(
    positive_text_wordcloud,
    "Positive Review Keywords",
    "Most frequent terms appearing in high-confidence positive reviews",
    "positive_wordcloud_5.png",
    positive_color_func
)

create_wordcloud(
    negative_text_wordcloud,
    "Negative Review Keywords",
    "Most frequent terms appearing in high-confidence negative reviews",
    "negative_wordcloud_5.png",
    negative_color_func
)
import re
def clean_review_for_summary(text):
    text = str(text)

    # Remove height patterns like 5'1, 5' 1", 5 ft 1
    text = re.sub(r"\b\d\s*'\s*\d{1,2}\b", "", text)
    text = re.sub(r"\b\d\s*ft\s*\d{0,2}\b", "", text, flags=re.IGNORECASE)

    # Remove weight patterns like 100#, 120 lbs
    text = re.sub(r"\b\d{2,3}\s*#\b", "", text)
    text = re.sub(r"\b\d{2,3}\s*(lbs|pounds)\b", "", text, flags=re.IGNORECASE)

    # Remove bra size patterns like 34B, 36C
    text = re.sub(r"\b\d{2}[a-d]{1,2}\b", "", text, flags=re.IGNORECASE)

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
# ==========================================
# 6. REVIEW + TEXT SUMMARIZATION MODULE
# ==========================================
print("Generating review and text summaries...")

import re
from collections import Counter

# ------------------------------------------
# 6.1 Text cleaning helpers
# ------------------------------------------

def split_sentences(text):
    text = str(text)
    return re.split(r'(?<=[.!?])\s+', text)


def basic_clean_text(text):
    text = str(text)

    # Remove personal measurements
    text = re.sub(r"\b\d\s*[\"']\s*\d{0,2}\b", "", text)          # 5'5, 5"5
    text = re.sub(r"\b\d{2,3}\s*#\b", "", text)                   # 100#
    text = re.sub(r"\b\d{2,3}\s*(lbs|pounds)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{2}[a-d]{1,2}\b", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_personal_measurement_sentence(sentence):
    """
    Removes sentences like:
    - I'm 5'5...
    - I ordered size S petite...
    - I typically wear XS...
    - If you're less busty...
    
    But keeps opinion sentences like:
    - I love the fabric.
    - I like the design.
    """
    sentence_lower = sentence.lower()

    personal_patterns = [
        r"\bi am\s+\d",
        r"\bi'm\s+\d",
        r"\bi am .*?\b\d\s*[\"']",
        r"\bi'm .*?\b\d\s*[\"']",
        r"\bi ordered\b",
        r"\bi typically wear\b",
        r"\bi usually wear\b",
        r"\bi normally wear\b",
        r"\bmy usual size\b",
        r"\bmy normal size\b",
        r"\bmy regular size\b",
        r"\bi sized up\b",
        r"\bi sized down\b",
        r"\bpetite\b",
        r"\bxs\b",
        r"\bxxs\b",
        r"\bcup\b",
        r"\bbusty\b",
        r"\bbust\b",
        r"\bwaist\b",
        r"\bhips\b",
        r"\b\d\s*[\"']\s*\d{0,2}\b",
        r"\b\d{2,3}\s*#\b",
        r"\b\d{2,3}\s*(lbs|pounds)\b",
        r"\b\d{2}[a-d]{1,2}\b"
    ]

    return any(re.search(pattern, sentence_lower) for pattern in personal_patterns)


def clean_text_for_text_summarization(text):
    """
    Keeps general opinion sentences, including first-person opinions like:
    - I love the fabric.
    - I like the fit.
    
    Removes personal measurement and sizing context like:
    - I'm 5'5...
    - I usually wear XS.
    """
    text = basic_clean_text(text)
    sentences = split_sentences(text)

    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) < 4:
            continue

        if is_personal_measurement_sentence(sentence):
            continue

        cleaned_sentences.append(sentence)

    cleaned_text = " ".join(cleaned_sentences)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def remove_negative_sentences_from_positive(text):
    negative_sentence_triggers = [
        "not flattering",
        "unflattering",
        "runs small",
        "too small",
        "too big",
        "terrible",
        "problem",
        "stuck",
        "poor quality",
        "bad quality",
        "cheap",
        "disappointed",
        "uncomfortable",
        "had to return",
        "returning",
        "does not fit",
        "did not fit",
        "odd",
        "tacky",
        "zipper",
        "zip",
        "buttons need",
        "fabric is terrible"
    ]

    sentences = split_sentences(text)
    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        sentence_lower = sentence.lower()

        if not sentence:
            continue

        if any(trigger in sentence_lower for trigger in negative_sentence_triggers):
            continue

        cleaned_sentences.append(sentence)

    return " ".join(cleaned_sentences).strip()


def remove_positive_sentences_from_negative(text):
    positive_sentence_triggers = [
        "nice choice",
        "beautiful",
        "love",
        "perfect",
        "pretty",
        "flattering",
        "comfortable",
        "wonderful",
        "excellent",
        "great",
        "gorgeous",
        "amazing",
        "looks good",
        "very cute",
        "so cute",
        "so pretty"
    ]

    sentences = split_sentences(text)
    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        sentence_lower = sentence.lower()

        if not sentence:
            continue

        if any(trigger in sentence_lower for trigger in positive_sentence_triggers):
            continue

        cleaned_sentences.append(sentence)

    return " ".join(cleaned_sentences).strip()


# ------------------------------------------
# 6.2 Customer review summary helpers
# ------------------------------------------

positive_aspects = {
    "soft or silky fabric": [
        "soft", "silky", "smooth", "nice fabric", "great fabric", "fabric feels good"
    ],
    "comfortable feel": [
        "comfortable", "comfy", "easy to wear", "wearable"
    ],
    "flattering fit": [
        "flattering", "fits well", "great fit", "perfect fit", "fit perfectly"
    ],
    "pretty design": [
        "pretty", "beautiful", "cute", "gorgeous", "lovely", "stylish", "elegant"
    ],
    "tulle or layered detail": [
        "tulle", "layer", "layers", "netting", "overlay"
    ],
    "good length": [
        "length", "not too long", "not too short"
    ],
    "overall satisfaction": [
        "love", "loved", "perfect", "wonderful", "amazing", "excellent", "great"
    ]
}

negative_aspects = {
    "sizing issues": [
        "runs small", "too small", "too big", "size up", "size down", "sizing"
    ],
    "poor fabric quality": [
        "poor quality", "bad quality", "cheap", "tacky", "thin", "terrible fabric",
        "fabric is terrible", "material looks cheap", "material feels cheap"
    ],
    "zipper problems": [
        "zipper", "zip", "stuck in the zip", "zip area"
    ],
    "button placement issues": [
        "button", "buttons", "buttons need"
    ],
    "unflattering cut": [
        "not flattering", "unflattering", "odd cut", "cut is odd", "weird cut"
    ],
    "see-through fabric": [
        "see-through", "transparent", "need a slip"
    ],
    "poor value for money": [
        "not worth", "for the cost", "for the price", "expensive", "overpriced"
    ],
    "return or disappointment": [
        "return", "returned", "returning", "disappointed", "disappointment"
    ]
}


def extract_aspects_from_text(text, aspect_dict):
    aspect_counts = Counter()
    text_lower = text.lower()

    for aspect, keywords in aspect_dict.items():
        for keyword in keywords:
            if keyword in text_lower:
                aspect_counts[aspect] += 1

    return aspect_counts


def build_customer_review_summary(aspect_counts, sentiment_label, max_aspects=5):
    aspects = [aspect for aspect, count in aspect_counts.most_common(max_aspects)]

    if sentiment_label == "Positive":
        if not aspects:
            return (
                "Customers generally praise the product for its overall appearance, "
                "comfort, and design."
            )

        if len(aspects) == 1:
            return f"Customers mainly praise the product for its {aspects[0]}."

        if len(aspects) == 2:
            return f"Customers mainly praise the product for its {aspects[0]} and {aspects[1]}."

        return (
            "Customers mainly praise the product for its "
            + ", ".join(aspects[:-1])
            + f", and {aspects[-1]}."
        )

    else:
        if not aspects:
            return (
                "Customers mainly complain about product quality, fit, and construction issues."
            )

        if len(aspects) == 1:
            return f"Customers mainly complain about {aspects[0]}."

        if len(aspects) == 2:
            return f"Customers mainly complain about {aspects[0]} and {aspects[1]}."

        return (
            "Customers mainly complain about "
            + ", ".join(aspects[:-1])
            + f", and {aspects[-1]}."
        )


# ------------------------------------------
# 6.3 Text summarization with BART
# ------------------------------------------

def generate_text_summary(text, sentiment_label):
    """
    Real text summarization.
    Allows first-person opinions like 'I love...'
    Removes personal measurement/sizing details before summarization.
    """
    words = text.strip().split()

    if len(words) < 15:
        return f"Not enough {sentiment_label.lower()} reviews to generate a meaningful text summary."

    # If text is short, do not force BART.
    # Return cleaned text directly.
    if len(words) < 60:
        return " ".join(words)

    truncated_text = " ".join(words[:800])
    input_word_count = len(truncated_text.split())

    max_len = min(120, max(35, int(input_word_count * 0.35)))
    min_len = min(35, max(15, int(input_word_count * 0.15)))

    try:
        summary_outputs = summarizer(
            truncated_text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

        summary_text = summary_outputs[0]["summary_text"]

        # Final cleanup: remove personal measurement sentences again
        summary_text = clean_text_for_text_summarization(summary_text)

        if sentiment_label == "Positive":
            summary_text = remove_negative_sentences_from_positive(summary_text)
        else:
            summary_text = remove_positive_sentences_from_negative(summary_text)

        if len(summary_text.split()) < 8:
            return " ".join(words[:50])

        return summary_text

    except Exception as e:
        return f"Could not generate text summary due to an error: {str(e)}"


# ------------------------------------------
# 6.4 Select high-confidence reviews
# ------------------------------------------

SUMMARY_CONFIDENCE_THRESHOLD = 0.85

positive_reviews_for_summary = df_sample[
    (df_sample["Sentiment"] == "Positive") &
    (df_sample["Confidence"] >= SUMMARY_CONFIDENCE_THRESHOLD)
][COLUMN_NAME].dropna().astype(str).tolist()

negative_reviews_for_summary = df_sample[
    (df_sample["Sentiment"] == "Negative") &
    (df_sample["Confidence"] >= SUMMARY_CONFIDENCE_THRESHOLD)
][COLUMN_NAME].dropna().astype(str).tolist()


# ------------------------------------------
# 6.5 Clean text for text summarization
# ------------------------------------------

positive_clean_texts = [
    clean_text_for_text_summarization(text)
    for text in positive_reviews_for_summary
]

negative_clean_texts = [
    clean_text_for_text_summarization(text)
    for text in negative_reviews_for_summary
]

positive_clean_texts = [
    remove_negative_sentences_from_positive(text)
    for text in positive_clean_texts
]

negative_clean_texts = [
    remove_positive_sentences_from_negative(text)
    for text in negative_clean_texts
]

positive_clean_texts = [
    text for text in positive_clean_texts
    if len(text.split()) >= 4
]

negative_clean_texts = [
    text for text in negative_clean_texts
    if len(text.split()) >= 4
]


positive_text_summary_input = " ".join(positive_clean_texts[:50])
negative_text_summary_input = " ".join(negative_clean_texts[:50])


# ------------------------------------------
# 6.6 Generate both summary types
# ------------------------------------------

positive_aspect_counts = extract_aspects_from_text(
    positive_text_summary_input,
    positive_aspects
)

negative_aspect_counts = extract_aspects_from_text(
    negative_text_summary_input,
    negative_aspects
)

positive_customer_review_summary = build_customer_review_summary(
    positive_aspect_counts,
    sentiment_label="Positive"
)

negative_customer_review_summary = build_customer_review_summary(
    negative_aspect_counts,
    sentiment_label="Negative"
)

positive_text_summary = generate_text_summary(
    positive_text_summary_input,
    sentiment_label="Positive"
)

negative_text_summary = generate_text_summary(
    negative_text_summary_input,
    sentiment_label="Negative"
)


print("Positive reviews selected:", len(positive_reviews_for_summary))
print("Negative reviews selected:", len(negative_reviews_for_summary))
print("Positive cleaned texts used:", len(positive_clean_texts))
print("Negative cleaned texts used:", len(negative_clean_texts))
print("Positive text summary word count:", len(positive_text_summary_input.split()))
print("Negative text summary word count:", len(negative_text_summary_input.split()))
print("Positive aspects:", positive_aspect_counts)
print("Negative aspects:", negative_aspect_counts)


# ------------------------------------------
# 6.7 Save report
# ------------------------------------------

report_path = os.path.join(OUTPUT_DIR, "executive_summary_report_5.txt")

report_content = f"""========================================================================
EXECUTIVE SUMMARY REPORT (AUTOMATED NLP INSIGHTS)
========================================================================
Generated Report Output

------------------------------------------------------------------------
🟢 POSITIVE TEXT SUMMARY:
------------------------------------------------------------------------
{positive_text_summary}

------------------------------------------------------------------------
🔴 NEGATIVE TEXT SUMMARY:
------------------------------------------------------------------------
{negative_text_summary}

------------------------------------------------------------------------
🟢 POSITIVE CUSTOMER REVIEW SUMMARY (What customers love):
------------------------------------------------------------------------
{positive_customer_review_summary}

------------------------------------------------------------------------
🔴 NEGATIVE CUSTOMER REVIEW SUMMARY (Pain points & areas of improvement):
------------------------------------------------------------------------
{negative_customer_review_summary}

========================================================================
End of Report
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"Saved executive summary report: {report_path}")
print("\nAll visualizations and text reports are ready. Check the outputs folder.")

test_review = "Customer service exceeded expectations."
print(f"Test Sonucu: {predict_sentiment_with_score(test_review)}")

