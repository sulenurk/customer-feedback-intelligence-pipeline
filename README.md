# Customer Review Sentiment Analysis & Summarization Pipeline

Python • PyTorch • Transformers • DistilBERT • BART • NLP • Hugging Face

## Project Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline for automated customer review analysis. The system classifies customer sentiment, generates visual insights, and produces both customer-level feedback summaries and text-based summaries from large collections of textual reviews.

The pipeline combines a fine-tuned DistilBERT sentiment classifier with confidence filtering, sentiment visualization, word cloud generation, customer review insight summarization, and BART-based text summarization.

The project is designed as a fully local workflow, allowing sentiment prediction, visualization, and report generation without relying on external inference APIs.

## Key Results

- Fine-tuned DistilBERT sentiment classifier
- 93.29% Accuracy
- 93.56% Precision
- 92.99% Recall
- 93.27% Macro F1 Score
- Confidence-based sentiment filtering
- Automated sentiment distribution visualization
- Positive and negative word cloud generation
- Customer review insight summarization
- BART-based text summarization
- Fully local inference workflow without external inference APIs

## Business Problem

Organizations receive large volumes of customer feedback through online reviews, surveys, and digital platforms. Manually analyzing this feedback is time-consuming, inconsistent, and difficult to scale.

The goal of this project is to automate customer review analysis by identifying sentiment polarity, visualizing sentiment patterns, and summarizing both positive feedback and customer pain points.

This pipeline helps transform raw customer reviews into structured insights that can support product improvement, customer experience analysis, and business reporting.

## Project Objectives

The main objectives of this project are:

- Build a transformer-based sentiment classification model for customer reviews
- Classify reviews as positive, negative, or neutral using confidence filtering
- Generate visual insights from classified reviews
- Identify frequently mentioned terms in positive and negative feedback
- Produce customer-level review summaries that describe recurring feedback themes
- Generate text summaries from sentiment-filtered reviews using BART
- Create a local and reproducible NLP workflow without external inference APIs

## Dataset

### Training Dataset

The sentiment classification model was fine-tuned using Amazon customer review data. This dataset provided labeled customer feedback for model training, validation, and performance evaluation.

**Dataset Size:** 400,000 labeled customer reviews

**Class Distribution:** 200,000 Negative / 200,000 Positive

**Target Column:** Review text

**Target Labels:** Binary sentiment classification

- `0` = Negative
- `1` = Positive

**Preprocessing Steps:**

- Tokenization using `distilbert-base-uncased`
- Sequence truncation and padding to a maximum length of 128 tokens
- Label column renaming from `label` to `labels` for Hugging Face Trainer compatibility
- Conversion of tokenized datasets into PyTorch tensor format

### Demonstration Dataset

To demonstrate the complete NLP workflow, a separate customer review dataset was used from Kaggle: **Women’s E-Commerce Clothing Reviews**.

This dataset contains women’s clothing e-commerce reviews written by customers. The data includes review text, ratings, recommendation information, product categories, and other review-related features. Since the dataset is based on real commercial data, it has been anonymized, and references to the company were replaced with “retailer”. :contentReference[oaicite:0]{index=0}

**Dataset Source:** Kaggle — Women’s E-Commerce Clothing Reviews  
**Dataset Author:** nicapotato  
**Dataset Link:** https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews  
**Dataset Size:** Approximately 23,000 customer reviews and ratings. :contentReference[oaicite:1]{index=1}

This dataset was not used during sentiment model training. Instead, it was used as inference input to demonstrate the full production-style pipeline, including:

- Sentiment classification
- Confidence-based sentiment filtering
- Sentiment distribution visualization
- Positive and negative word cloud generation
- Customer review insight summarization
- Text summarization from sentiment-filtered reviews

This separation allows the project to distinguish between model development and downstream business intelligence workflows.

## Model Performance

The final fine-tuned DistilBERT model was evaluated on a held-out validation split from the labeled Amazon customer review dataset.

| Metric         | Score      |
| -------------- | ---------- |
| Accuracy       | **93.29%** |
| Precision      | **93.56%** |
| Recall         | **92.99%** |
| Macro F1 Score | **93.27%** |

The close alignment between precision, recall, and macro F1-score indicates that the classifier performs consistently across both positive and negative sentiment classes.

The model achieved strong performance while remaining lightweight enough for local inference, making it suitable for batch customer review analysis without relying on external inference APIs.

Confusion Matrix
The confusion matrix below shows the classification performance of the fine-tuned DistilBERT model on the validation dataset.

<img width="1536" height="1024" alt="confusion_matrix" src="https://github.com/user-attachments/assets/36273ee5-460f-47aa-bb12-8b1ee5d3e489" />

## Model Architecture

The project uses a multi-stage transformer-based NLP architecture. Sentiment classification, visualization, customer review summarization, and text summarization are handled as separate components within the same pipeline.

### 1. Sentiment Classification

The sentiment analysis component is based on a fine-tuned DistilBERT model.

**Base Model:** `distilbert-base-uncased`

DistilBERT was selected because it provides a strong balance between classification performance and inference efficiency. This makes it suitable for local deployment and batch processing without external API calls.

The classifier predicts whether a customer review expresses positive or negative sentiment. A confidence threshold is then applied to reduce noisy predictions and assign uncertain cases to a neutral category.

### 2. Confidence Filtering

The model outputs class probabilities for negative and positive sentiment. Reviews with low prediction confidence are labeled as neutral.

This filtering step improves the quality of downstream outputs by reducing sentiment leakage between positive and negative review groups.

Confidence filtering supports:

- More reliable sentiment distribution analysis
- Cleaner positive and negative word clouds
- More accurate customer review summaries
- Better text summarization inputs


Ama GitHub README içinde iç içe code block bazen karışmasın diye en güvenli hali şöyle:

### 3. Customer Review Insight Summarization

The pipeline includes a customer review insight summarization module designed to convert individual review comments into broader customer-level feedback.

Instead of directly copying raw review sentences, this module focuses on recurring product-related themes such as:

- Fit
- Comfort
- Fabric quality
- Design
- Sizing issues
- Zipper or button problems
- Overall satisfaction

This allows the report to describe what customers generally like or dislike about a product.

Example input:
I love the fabric and the dress looks flattering.
Example output:
Customers mainly praise the product for its soft fabric, flattering fit, and pretty design.

## Pipeline Workflow

The pipeline follows a structured end-to-end workflow:

1. Load customer review data.
2. Preprocess review text.
3. Run sentiment classification using the fine-tuned DistilBERT model.
4. Convert model logits into sentiment probabilities.
5. Apply confidence-based filtering.
6. Separate reviews into positive, negative, and neutral groups.
7. Generate sentiment distribution visualization.
8. Generate positive and negative word clouds.
9. Clean review text for summarization.
10. Generate customer review insight summaries.
11. Generate BART-based text summaries.
12. Export visualizations and summary reports locally.

## Pipeline Outputs

Each execution of the pipeline automatically generates a collection of analytical assets inside the local `./outputs/` directory.

### 1. Sentiment Distribution Analysis

**Output:** `sentiment_distribution.png`

A sentiment distribution chart showing the percentage and count of positive, negative, and neutral reviews.

**Purpose:**

* Measure overall customer sentiment.
* Identify sentiment trends at a glance.
* Support high-level business reporting.

---
Note: The visual outputs shown in this repository are generated from the synthetic customer feedback dataset used for pipeline demonstration, while the sentiment classifier itself was trained on Amazon review data.

### 2. Positive Sentiment Word Cloud

**Output:** `positive_wordcloud.png`

A word cloud highlighting the most frequently occurring terms in positive customer reviews.

**Purpose:**

* Identify product strengths.
* Reveal features that customers value most.
* Support marketing and product positioning decisions.

---

### 3. Negative Sentiment Word Cloud

**Output:** `negative_wordcloud.png`

A word cloud generated from negative customer feedback after stop-word filtering and sentiment separation.

**Purpose:**

* Identify recurring customer complaints.
* Detect product weaknesses and pain points.
* Support product improvement initiatives.

---

### 4. Executive Summary Report

**Output:** `executive_summary_report_5.txt`

The executive summary report contains both customer-level feedback summaries and text-based summaries.

The report includes:

- Positive Customer Review Summary
- Positive Text Summary
- Negative Customer Review Summary
- Negative Text Summary

**Purpose:**

- Summarize large volumes of customer reviews
- Highlight what customers generally like
- Identify recurring product complaints
- Preserve useful review-style language when needed
- Reduce manual review effort
---

### 5. Fine-Tuned Sentiment Model

**Output:** `./my_sentiment_model/`

A locally stored DistilBERT model containing the fine-tuned weights, tokenizer files, and configuration required for offline inference.

**Purpose:**

* Enable low-latency local predictions.
* Eliminate dependency on external APIs.
* Support reproducible deployment and future experimentation.

## Development Journey

The project evolved through several development iterations. Early versions focused on testing the end-to-end inference pipeline with generated review samples, while later versions improved the workflow using a real-world Kaggle dataset and more advanced summarization logic.

| Version | Main Focus | Outcome |
| ------- | ---------- | ------- |
| V1 | Initial inference pipeline using GPT-generated random customer reviews | Established the first working version of the sentiment classification and reporting workflow |
| V2 | Sentiment visualization on generated review samples | Added sentiment distribution charts and improved the visual reporting structure |
| V3 | Word cloud generation for positive and negative reviews | Added keyword-level insight extraction from sentiment-separated review groups |
| V4 | First executive summary generation using BART | Tested automated positive and negative review summarization on generated review data |
| V5 | Production-style local pipeline using generated reviews | Combined sentiment classification, visualization, word clouds, and BART-based summarization into a single local workflow |
| V6 | Migration to the Kaggle Women’s E-Commerce Clothing Reviews dataset | Confirmed that positive and negative summarization logic worked on real review data, but identified the need to remove personal sizing and body-measurement details from summaries |
| V7 | Summary cleaning and personal-detail filtering | Added preprocessing rules to remove personal measurements, sizing context, and overly individual review details before summarization |
| V8 | Customer review insight summarization | Added a separate customer-level summary module that converts recurring review themes into business-style insights instead of copying individual review sentences |
| V9 | Dual-summary reporting | Added both Customer Review Summary and Text Summary outputs, allowing the report to provide structured business insights while still preserving useful review-style text when appropriate |

## Future Improvements

### 1. Aspect-Based Sentiment Analysis

The current pipeline classifies each review into an overall sentiment category. A future improvement would be to identify sentiment toward specific product aspects such as fit, fabric quality, comfort, sizing, design, zipper quality, and value for money.

This would allow the system to answer more detailed business questions, such as whether customers like the design but complain about sizing or fabric quality.

### 2. Improved Summarization with Instruction-Tuned Models

The current text summarization module uses `facebook/bart-large-cnn`, which is effective for general summarization but is not specifically instruction-tuned for customer feedback analysis.

Future versions could evaluate instruction-tuned language models to generate more natural, structured, and context-aware summaries while better avoiding personal details and irrelevant review fragments.

### 3. Automated Theme Extraction

The customer review insight summary currently relies on predefined aspect keywords. A future version could use clustering, topic modeling, or embedding-based methods to automatically discover recurring themes from reviews.

This would make the pipeline more flexible across different product categories without manually defining aspect dictionaries.

### 4. Multi-Class Sentiment Classification

The current classifier is based on binary sentiment labels with confidence-based neutral filtering. A future improvement would be to train a dedicated multi-class sentiment model for positive, negative, neutral, and mixed reviews.

This could improve the handling of reviews that contain both praise and complaints.

### 5. Model Optimization for Faster Inference

The fine-tuned DistilBERT model is already suitable for local inference, but performance could be improved further through optimization techniques such as ONNX conversion, quantization, or TorchScript export.

This would make the pipeline faster and more efficient for larger review datasets.

### 6. Interactive Dashboard

The current pipeline exports static charts, word clouds, and text reports. A future version could integrate the outputs into an interactive dashboard using Streamlit, Dash, or a web-based interface.

This would allow users to explore sentiment trends, filter reviews by category, inspect summaries, and compare positive and negative feedback more easily.

### 7. Deployment as a Reusable NLP Tool

The project could be packaged as a reusable command-line tool or lightweight web application. This would make it easier to run the pipeline on different review datasets with minimal configuration changes.

## Histogram Screenshot
<img width="2257" height="1296" alt="image" src="https://github.com/user-attachments/assets/b8a3100c-a5f9-46c5-8f9f-9fc9af6b8c2d" />


## Positive Word Cloud Screenshot
<img width="2264" height="1780" alt="image" src="https://github.com/user-attachments/assets/73535aed-9f15-4ac9-a370-5d96dbb4d0e5" />

## Sample Executive Summary 
------------------------------------------------------------------------
### POSITIVE TEXT SUMMARY:
------------------------------------------------------------------------
I love this jumpsuit. It's fun, flirty, and fabulous! every time i wear it, i get nothing but great compliments! This shirt is very flattering to all due to the adjustable front tie. It is the perfect length to wear with leggings and it is sleeveless so it pairs well with any cardigan. I'm upset because for the price of the dress, i thought it was embroidered! no, that is a print on the fabric.

------------------------------------------------------------------------
### NEGATIVE TEXT SUMMARY:
------------------------------------------------------------------------
The fabric feels cheap, like it will snag easily and will stretch out quickly. Bought a large, could barely pull up over my butt. It's cute but if your not a stick figure, this is not the suit for you.

------------------------------------------------------------------------
### POSITIVE CUSTOMER REVIEW SUMMARY (What customers love):
------------------------------------------------------------------------
Customers mainly praise the product for its overall satisfaction, pretty design, tulle or layered detail, soft or silky fabric, and comfortable feel.

------------------------------------------------------------------------
### NEGATIVE CUSTOMER REVIEW SUMMARY (Pain points & areas of improvement):
------------------------------------------------------------------------
Customers mainly complain about return or disappointment, sizing issues, poor fabric quality, zipper problems, and button placement issues.

