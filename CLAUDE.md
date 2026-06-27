# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Corpus linguistics analysis of 50,000 ChatGPT Google Play reviews. The goal is to identify lexical and discursive differences between positive and negative reviews using rule-based NLP methods (no ML). A Streamlit app provides interactive TF-IDF similarity search over the reviews.

## Running the App

```bash
streamlit run app.py
```

The app loads `data/chatgpt_reviews_clean_balanced.csv` and serves a TF-IDF cosine similarity search UI.

## Data Files

| File | Description |
|------|-------------|
| `data/chatgpt_reviews_raw.csv` | 50K raw reviews from Google Play (all languages) |
| `data/chatgpt_reviews_clean_balanced.csv` | Balanced English-only corpus (score 3 excluded) |
| `data/chatgpt_reviews_clean_balanced_v2.csv` | Alternative balanced version |
| `data/reviews_with_labels.csv` | Reviews with positive/negative labels |

Raw data columns: `reviewId`, `userName`, `userImage`, `content`, `score`, `thumbsUpCount`, `reviewCreatedVersion`, `at`, `replyContent`, `repliedAt`, `appVersion`.

## Notebooks (scripts/)

Run notebooks from the `scripts/` directory — data paths use `../data/`.

| Notebook | Purpose |
|----------|---------|
| `getting_corpora.ipynb` | Scrapes reviews via `google-play-scraper` |
| `corpora_prep.ipynb` | Language detection, labeling, tokenization, lemmatization, balancing |
| `corpora_analysis.ipynb` | Frequency analysis, bigrams, review length comparison |
| `similarity.ipynb` | TF-IDF vectorization and cosine similarity exploration |
| `score_by_version.ipynb` | Review counts and average scores grouped by date/app version |

## Key Design Decisions

- **No ML**: classification and analysis are purely rule-based and corpus-linguistic.
- **English-only analysis**: `langdetect` filters to `language == 'en'` before any NLP. Detection is unreliable on short texts.
- **Balancing**: negative class (score 1–2) is smaller; positive (score 4–5) is randomly downsampled to match. Score 3 is excluded.
- **TF-IDF model** in `app.py` is built on the balanced corpus (`chatgpt_reviews_clean_balanced.csv`), not the raw one.
- The `.venv` uses Python 3.13; notebooks use the system Python 3.10 (libraries like `spacy`, `langdetect`, `nltk` are installed there).
