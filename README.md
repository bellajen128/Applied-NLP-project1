# Slangify Project

自動將句子中的詞替換成俚語的 NLP 系統。

## 📁 專案結構
```
NLP_latest/
├── data/                          # 資料檔
│   ├── raw/                       # 原始資料 (ud_2015-2025.csv)
│   ├── slang_raw_combined.csv     # 合併原始資料 (59163 條)
│   └── slang_clean_final.csv      # 清理後資料 (9173 條) ⭐
│
├── models/                        # 訓練好的模型
│   └── best_slang_bert_classifier.pt  # BERT Classifier (F1: 0.9858) ⭐
│
├── scripts/                       # 主要腳本
│   ├── official_preprocessing.py  # 資料清理 pipeline
│   ├── train_classifier.py        # 訓練 BERT Classifier
│   └── test_baseline_clean.py     # 測試 Baseline 系統
│
├── training/                      # 訓練相關檔案
│   ├── training_data_clean.json   # BERT 訓練資料 (6000 樣本) ⭐
│   └── training_history_*.json    # 訓練歷史記錄
│
└── archive/                       # 舊版/測試檔案
    ├── processed_slang_data.csv   # 舊版資料
    ├── training_data.json         # 舊版訓練資料
    ├── test_*.py                  # 各種測試腳本
    └── ...
```

## 🎯 Baseline 系統架構
```
輸入句子
    ↓
[1] 關鍵詞提取 (spaCy)
    → 提取可替換詞 + 詞性標註
    → 過濾 SKIP_WORDS (黑名單功能詞)
    → 排序: ADJ > VERB > NOUN
    ↓
[2] FAISS 檢索 (預訓練 all-MiniLM-L6-v2)
    → 為每個關鍵詞搜尋候選
    → 過濾垃圾詞 (NER, 真實詞檢查, 統計特徵)
    ↓
[3] BERT Classifier 評分 (訓練好的 DistilBERT)
    → 判斷 (sentence, slang) 配對適配性
    ↓
[4] Combined Score
    → 0.35 × FAISS + 0.65 × BERT + Bonus
    → Bonus: POS + Match + Popularity (白名單 +0.25)
    ↓
[5] 選擇最佳候選並替換
```

## 🚀 快速開始

### 1. 資料清理
```bash
python scripts/official_preprocessing.py \
    --input data/slang_raw_combined.csv \
    --output data/slang_clean_final.csv \
    --min_quality 4
```

**輸出：** 9173 條高品質 slang

---

### 2. 訓練 BERT Classifier
```bash
# 先在 Jupyter 生成訓練資料 (training_data_clean.json)
# 然後訓練：

python scripts/train_classifier.py \
    --data training/training_data_clean.json \
    --epochs 3 \
    --batch_size 16
```

**輸出：** models/best_slang_bert_classifier.pt (Val F1: 0.9858)

---

### 3. 測試系統
```bash
python scripts/test_baseline_clean.py
```

**測試結果：**
- new → fresh ⭐
- amazing → lit af ⭐
- upset → salty ⭐
- leave → bounce ⭐
- suspicious → sus ⭐

---

## 📊 系統效能

| 指標 | 數值 |
|------|------|
| 資料量 | 9173 條乾淨 slang |
| BERT Val F1 | 0.9858 |
| 測試成功率 | 100% (5/5) |
| 推理速度 | ~0.5s per query |

---

## 🔧 配置參數

### FAISS Retrieval
- `k_per_keyword`: 5 (每個關鍵詞取 5 個候選)
- `min_faiss_score`: 0.25

### BERT Classifier
- `alpha`: 0.35 (FAISS vs BERT 權重)
- `conf_threshold`: 0.55

### Bonus Scores
- POS: ADJ +0.15, NOUN +0.05, VERB +0.0
- Match: Definition 匹配 +0.15
- Popularity: 白名單 +0.25

---

## 📝 重要檔案說明

### 資料檔
- `slang_clean_final.csv` - **最終使用的乾淨資料** ⭐
- `slang_raw_combined.csv` - 原始合併資料（備份用）

### 模型檔
- `best_slang_bert_classifier.pt` - **訓練好的 BERT Classifier** ⭐

### 訓練資料
- `training_data_clean.json` - **BERT 訓練資料** ⭐
  - 3000 正樣本
  - 1500 Hard Negative
  - 1500 Easy Negative

---

## 🎯 下一步開發

- [ ] 互動式多詞替換模組
- [ ] Streamlit UI 介面
- [ ] 部署到雲端

---

## 📚 參考文件

- Preprocessing: `scripts/official_preprocessing.py`
- Training: `scripts/train_classifier.py`
- Testing: `scripts/test_baseline_clean.py`

