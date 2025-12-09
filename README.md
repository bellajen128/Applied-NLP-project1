# Slangify - Interactive Slang Generation System

An NLP-powered system that automatically converts formal language into contemporary slang while preserving semantic meaning. Built with dual-model architecture (FAISS + BERT) for real-time, context-aware slang suggestions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]( https://applied-nlp-project1-rh25dgdawpw9evd6fbckoe.streamlit.app/)

## Features

- **Automatic Mode**: Single-click slang replacement with best candidate
- **Interactive Mode**: Multi-word replacement with top-k suggestions
- **Real-time Performance**: 40ms inference time per sentence
- **High Accuracy**: 97.89% F1 on test set
- **9,173 Curated Slang**: Filtered from 59K+ Urban Dictionary entries (2015-2025)

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/ bellajen128/Applied-NLP-project1.git
cd Applied-NLP-project1

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_md

# Run Streamlit app
streamlit run app.py
```

### Usage Example
```python
from scripts.slangify_core import SlangifySystem

# Initialize system
system = SlangifySystem()

# Automatic mode
sentence = "He likes to show off his new car."
result, best = system.slangify(sentence)
print(result)  # "He likes to show off his fresh car."

# Interactive mode
tokens = system.analyze_sentence(sentence)
suggestions = system.get_suggestions(sentence, "new", "ADJ", 6, top_k=3)
# Returns: [{'slang': 'fresh', 'score': 0.74, ...}, ...]
```

## System Architecture
```
Input Sentence
    ↓
┌─────────────────────────┐
│  Keyword Extraction     │
│  (spaCy POS + Lemma)    │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  FAISS Retrieval        │
│  (Sentence-BERT)        │
│  9,173 → ~30 candidates │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  BERT Classifier        │
│  (DistilBERT)           │
│  Context-aware ranking  │
└─────────────────────────┘
    ↓
Combined Score = 0.35×FAISS + 0.65×BERT + Bonuses
    ↓
Best Slang Replacement
```

## Technical Stack

**NLP Technologies:**
- **spaCy** (3.8.2): POS tagging, lemmatization, dependency parsing, NER
- **Sentence-BERT** (all-MiniLM-L6-v2): Semantic embeddings for retrieval
- **DistilBERT**: Binary classifier for slang-sentence pairing (66M parameters)
- **FAISS**: Fast vector similarity search (Facebook AI)

**Frameworks:**
- PyTorch 2.5.1
- Transformers 4.46.3
- Streamlit 1.40.0

## Project Structure
```
Applied-NLP-project1/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
│
├── data/
│   └── slang_clean_final.csv   # 9,173 curated slang entries
│
├── models/
│   └── best_slang_bert_classifier.pt  # Trained BERT (Test F1: 0.9789)
│
├── scripts/
│   ├── slangify_core.py           # Core system (all functions)
│   ├── official_preprocessing.py  # 6-stage data cleaning pipeline
│   ├── train_classifier_with_test.py  # BERT training
│   ├── evaluate_comprehensive.py  # Evaluation metrics
│   └── create_visualizations.py   # Generate charts
│
└── training/
    └── training_data_clean.json   # 6,000 self-supervised samples
```

## 🔬 Data Processing Pipeline

**From 59,163 to 9,173 entries (15.5% retention):**

1. **Filter 0**: Basic structure (length, spaces, numbers)
2. **Filter 1**: Profanity removal (393 words from GitHub)
3. **Filter 2**: NER filtering (person names via spaCy)
4. **Filter 3**: Real word verification (PyEnchant dictionary)
5. **Filter 4**: Definition quality (length, structure, common words)
6. **Filter 5**: Statistical features (vowel ratio, repetition)

**Whitelist Protection:** 150+ popular slang always retained

## Training Strategy

**Self-Supervised Learning** (no manual annotation):

- **Positive Samples (3,000)**: Keyword from definition → template sentence
  - Example: "upset" from "salty : angry" → "He is very upset." (label=1)
  
- **Hard Negatives (1,500)**: Same POS but unrelated
  - Example: "He is very upset." + "car : vehicle" (label=0)
  
- **Easy Negatives (1,500)**: Random pairings
  - Example: "He is happy." + "bounce : leave" (label=0)

**Training:**
- Split: 70% train / 15% val / 15% test
- 3 epochs, AdamW optimizer (lr=2e-5)
- Final Test F1: **0.9789**

## 📈 Evaluation Results

### Controlled Test Set (10 sentences)

| Metric | Value | Target |
|--------|-------|--------|
| Replacement Rate | 90.0% | >80% ✅ |
| Whitelist Coverage | 46.7% | >40% ✅ |
| Semantic Preservation | 0.698 | 0.65-0.85 ✅ |
| Inference Time | 0.04s | <1s ✅ |

### Real-World Test Set (100 sentences)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Replacement Rate | 60.0% | Many words lack slang |
| Whitelist Coverage | 11.1% | Need broader coverage |
| Semantic Preservation | 0.821 | Excellent retention |
| BERT Confidence | 0.890 | High certainty |
| Perplexity Delta | +572.7 | Style change (expected) |

## Example Outputs
```
Input:  "He likes to show off his new car."
Output: "He likes to show off his fresh car."
Slang:  new → fresh (score: 0.74, whitelist: ⭐)

Input:  "That party was really amazing last night."
Output: "That party was really lit af last night."
Slang:  amazing → lit af (score: 1.25, whitelist: ⭐)

Input:  "She is upset because she lost the game."
Output: "She is salty because she lost the game."
Slang:  upset → salty (score: 1.39, whitelist: ⭐)

Input:  "I need to leave soon."
Output: "I need to bounce soon."
Slang:  leave → bounce (score: 1.22, whitelist: ⭐)
```

## Key Design Decisions

### 1. Why Dual-Model Architecture?

**FAISS (Retrieval):**
- Fast: 0.01s to search 9,173 entries
- Reduces search space to ~30 candidates

**BERT (Re-ranking):**
- Accurate: Context-aware scoring
- Only evaluates top candidates

**Result:** 10x faster than BERT-only, more accurate than FAISS-only

### 2. Why Self-Supervised Learning?

**Alternative:** Manual annotation
- Would require labeling thousands of (sentence, slang) pairs
- High cost, time-consuming

**Our Approach:** Auto-generate labels
- Positive: Keyword in definition → good pair
- Negative: Keyword not in definition → bad pair
- Achieved 97.89% F1 with zero annotation

### 3. Why Whitelist Bonus?

**Problem:** BERT training data has bias
- "fresh" only has 11 training samples
- BERT gives fresh low score (0.03)

**Solution:** +0.25 bonus for 150+ popular slang
- Ensures common slang ranks high
- Compensates for data limitations

## Technical Implementation

### Combined Score Formula
```
Combined = (α × FAISS + (1-α) × BERT + POS + Popularity) / 1.40

Where:
- α = 0.35 (35% FAISS, 65% BERT)
- POS Bonus: ADJ +0.15, NOUN +0.05, VERB +0.0
- Popularity: Whitelist +0.25, Regular +0.05
- Normalized to [0, 1]
```

### Filtering Rules

**Candidate filtering (10+ rules):**
- Remove functional words (SKIP_WORDS: is, the, very, etc.)
- Remove person names (spaCy NER)
- Remove uppercase words (except acronyms ≤5 chars)
- Check vowel ratio (15-50%)
- Remove keyboard sequences (qwer, asdf)
- Avoid original word similarity

## Performance Metrics

- **BERT Test F1**: 0.9789 (excellent model accuracy)
- **Replacement Rate**: 60-90% (depends on text complexity)
- **Semantic Preservation**: 0.82 (good meaning retention)
- **Inference Speed**: 0.04s (real-time)
- **Model Size**: 767MB (DistilBERT checkpoint)

## Limitations

1. **Training Data Bias**
   - Some words (e.g., "fresh") have limited training samples
   - BERT gives low scores, compensated by whitelist

2. **English Only**
   - Data source limitation
   - No multilingual support

3. **Static Database**
   - Based on 2015-2025 data
   - New emerging slang not included

4. **Moderate Real-World Coverage**
   - 60% replacement rate on diverse corpus
   - Many formal/technical words lack slang alternatives

## Future Improvements

- [ ] Expand training data with diverse sentence templates
- [ ] Add more fresh samples to fix BERT bias
- [ ] Multi-language support
- [ ] Personalization by user demographics
- [ ] Context-aware replacement (full dialogue)
- [ ] Continuous updates with emerging slang

## Citation
```bibtex
@misc{slangify2025,
  title={Slangify: Interactive Slang Generation with Dual-Model Architecture},
  author={Bella Jen},
  year={2025},
  publisher={GitHub},
  url={https://github.com/ bellajen128/Applied-NLP-project1}
}
```

## Acknowledgments

- Urban Dictionary community for slang data (2015-2025)
- Hugging Face for pre-trained models
- spaCy, FAISS, Sentence-Transformers teams

## Contact

For questions or feedback: jen.che @northeastern.edu

---

