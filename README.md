Python • PyTorch • Transformers • DistilBERT • BART • NLP • Hugging Face

## Project Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline for automated customer review analysis. The system classifies customer sentiment, generates visual insights, and produces executive-level summaries from large collections of textual feedback.

The solution combines a fine-tuned DistilBERT sentiment classifier with automated visualization and abstractive summarization components, creating a fully local workflow that operates without external inference APIs.

## Key Results

- Fine-tuned DistilBERT sentiment classifier
- 93.29% Accuracy
- 93.29% Macro F1 Score
- Automated sentiment visualization pipeline
- Automated executive summary generation using BART
- Fully local deployment without external inference APIs

## Business Problem

Organizations receive large volumes of customer feedback through reviews, surveys, and digital platforms. Manually analyzing this information is time-consuming, inconsistent, and difficult to scale.

The goal of this project is to automate sentiment analysis and feedback summarization, enabling faster identification of customer pain points, product strengths, and emerging trends while reducing manual review effort.

## Dataset

### Training Dataset

The sentiment classification model was fine-tuned using Amazon customer review data. This dataset provided labeled customer feedback used for model training, validation, and performance evaluation.

**Dataset Size:** 400,000 labeled customer reviews

**Class Distribution:** 200,000 Negative / 200,000 Positive (balanced)

**Target Column:** Review (unstructured customer feedback)

**Target Labels:** Binary sentiment classification (0 = Negative, 1 = Positive)

**Preprocessing Steps:**
- Tokenization using `distilbert-base-uncased`
- Sequence truncation and padding to a maximum length of 128 tokens
- Label column renaming from `label` to `labels` for Hugging Face Trainer compatibility
- Conversion of tokenized datasets into PyTorch tensor format

### Demonstration Dataset

To demonstrate the complete NLP workflow, a separate synthetic customer feedback dataset was generated using GPT-assisted prompting.

This dataset was not used during model training. Instead, it served as inference input for the production pipeline, including:

* Sentiment classification
* Sentiment distribution analysis
* Word cloud generation
* Executive summary generation

This separation allows the project to distinguish between model development and real-world business intelligence workflows.

## Results

The final fine-tuned DistilBERT model was evaluated on a held-out validation dataset containing **400,000 labeled customer reviews**.

| Metric         | Score      |
| -------------- | ---------- |
| Accuracy       | **93.29%** |
| Precision      | **93.56%** |
| Recall         | **92.99%** |
| Macro F1 Score | **93.27%** |

The close alignment between precision, recall, and F1-score indicates a well-balanced classifier with no significant bias toward either sentiment class.

Confusion Matrix

<img width="1536" height="1024" alt="confusion_matrix" src="https://github.com/user-attachments/assets/36273ee5-460f-47aa-bb12-8b1ee5d3e489" />

## Model Architecture

The pipeline uses a two-stage transformer architecture, separating sentiment classification from text generation tasks.

### Sentiment Classification

The sentiment analysis component is based on a fine-tuned DistilBERT (`distilbert-base-uncased`) model. DistilBERT was selected as a balance between classification performance and inference speed, making it suitable for local deployment without relying on external APIs.

The model was fine-tuned on Amazon customer review data to classify customer feedback and support downstream business intelligence tasks.

### Abstractive Summarization

To transform large volumes of reviews into concise business insights, the pipeline utilizes BART Large CNN (`facebook/bart-large-cnn`).

The summarization module processes sentiment-filtered review groups and generates structured executive summaries, allowing users to quickly identify customer pain points and positive product feedback.

### Pipeline Workflow

1. Load and preprocess customer reviews.
2. Perform sentiment classification using the fine-tuned DistilBERT model.
3. Apply confidence filtering and sentiment grouping.
4. Generate visual analytics assets (distribution charts and word clouds).
5. Create executive summaries using BART.
6. Export all outputs locally for reporting and analysis.

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

**Output:** `executive_summary_report.txt`

An automatically generated business-oriented summary created using the BART summarization model.

**Purpose:**

* Condense large volumes of reviews into actionable insights.
* Highlight positive trends and critical issues.
* Reduce manual review effort.

---

### 5. Fine-Tuned Sentiment Model

**Output:** `./my_sentiment_model/`

A locally stored DistilBERT model containing the fine-tuned weights, tokenizer files, and configuration required for offline inference.

**Purpose:**

* Enable low-latency local predictions.
* Eliminate dependency on external APIs.
* Support reproducible deployment and future experimentation.

## Development Journey

The project evolved through multiple training and evaluation cycles, with each iteration addressing limitations observed in the previous version.

| Version | Main Objective                                                              | Outcome                                                                           |
| ------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| V1      | Establish a baseline sentiment classifier using a limited training schedule | Revealed underfitting and poor handling of contextual expressions                 |
| V2      | Introduce persistent checkpointing and recovery mechanisms                  | Improved training stability and reproducibility                                   |
| V3      | Extend fine-tuning duration and increase training exposure                  | Improved contextual understanding and reduced classification errors               |
| V4      | Complete full model convergence and evaluation                              | Achieved 93.29% accuracy and 93.29% macro F1-score                                |
| V5      | Build a production-ready inference pipeline                                 | Added confidence filtering, visualization generation, and automated summarization |

## Future Improvements

Several enhancements could further improve the pipeline:

### 1. Aspect-Based Sentiment Analysis (ABSA)

Extend the sentiment classifier to identify sentiment toward specific product aspects (e.g., delivery, battery life, customer service) rather than assigning a single sentiment label to an entire review.

### 2. Model Optimization

Reduce inference time and deployment costs through model compression techniques such as ONNX conversion and quantization.

### 3. Advanced Summarization

Evaluate larger language models to generate more detailed and context-aware business summaries from customer feedback.

### 4. Interactive Dashboard

Integrate the pipeline with a web-based dashboard to allow real-time exploration of sentiment trends, word clouds, and executive reports.

## Histogram Screenshot
<img width="2244" height="1296" alt="sentiment_distribution_5" src="https://github.com/user-attachments/assets/6308e724-fc4b-4e97-b009-0a74e52b6b6d" />

## Positive Word Cloud Screenshot
<img width="2264" height="1780" alt="positive_wordcloud_5" src="https://github.com/user-attachments/assets/f172c24c-d168-499f-80c0-a6c37aa0f0fb" />

## Sample Executive Summary 
### Positive Feedback Summary
> Amazing battery life that easily lasts two full days Excellent build quality and the device feels premium Fast shipping and the package arrived earlier than expected The screen is bright sharp and beautiful Performance is smooth even with many apps open The product exceeded all my expectations Battery charging speed is incredibly fast.
------------------------------------------------------------------------
### Negative Feedback Summary
> The product failed to meet basic expectations. Battery life is terrible and drains within hours. The materials feel fragile and poorly made. The device freezes frequently during use. Battery degradation happened unusually fast. Overall this was a very disappointing purchase

