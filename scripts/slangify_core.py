"""
Slangify Core Module
整合所有核心功能，可被 Streamlit 或其他應用 import

功能：
1. Baseline Pipeline (單一最佳替換)
2. Interactive Mode (多詞互動替換)
"""

import torch
import faiss
import pandas as pd
import spacy
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path # 引入 Pathlib 進行路徑處理

# ============================================
# 配置
# ============================================

SKIP_WORDS = {
    'soon', 'now', 'then', 'later', 'today', 'tomorrow', 'yesterday',
    'always', 'never', 'often', 'sometimes', 'already', 'still', 'recently',
    'need', 'want', 'will', 'would', 'could', 'should', 'must', 'can', 'may', 'might',
    'very', 'really', 'just', 'also', 'too', 'even', 'only', 'quite', 'pretty',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'get', 'got', 'make', 'made',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their',
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'of', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'in', 'out',
    'because', 'but', 'and', 'or', 'so', 'if', 'when', 'while',
}

NAME_BLACKLIST = {
    'axel', 'nick', 'andreas', 'qwer', 'seehia', 'dad', 'trust',
    'natalie guest', 'kilt custard', 'squeet', 'gekyume'
}

POPULAR_SLANG = {
    'lit', 'fire', 'dope', 'sick', 'fresh', 'salty', 'sus', 'bounce', 'dip',
    'lit af', 'sus af', 'dead', 'fr', 'lowkey', 'highkey', 'bet', 'cap',
    'no cap', 'vibe', 'mood', 'slap', 'cringe', 'based', 'yeet', 'flex',
    'shady', 'throw', 'goat', 'bussin', 'finna', 'bruh', 'savage', 'stan',
    'tea', 'snatched', 'drip', 'woke', 'hype', 'legit', 'ez', 'gg'
}

POS_PRIORITY = {
    'ADJ': 0.15,
    'ADV': 0.10,
    'NOUN': 0.05,
    'VERB': 0.0
}


# ============================================
# Slangify System Class
# ============================================

class SlangifySystem:
    """
    完整的 Slangify 系統
    
    功能：
    1. Baseline mode: 自動替換最佳候選
    2. Interactive mode: 多詞互動替換
    """
    
    def __init__(self, 
                 data_rel_path='data/slang_clean_final.csv',
                 model_rel_path='models/best_slang_bert_classifier.pt',
                 use_gpu=True):
        """
        初始化系統
        
        Parameters:
        -----------
        data_rel_path : str
            Slang 資料檔的相對路徑 (相對於專案根目錄)
        model_rel_path : str
            訓練好的 BERT Classifier 的相對路徑
        use_gpu : bool
            是否使用 GPU
        """
        print("="*60)
        print("Initializing Slangify System")
        print("="*60)
        
        # --- 檔案路徑修正 (使用 Pathlib) ---
        # 獲取當前腳本的絕對路徑
        current_script_path = Path(__file__).resolve()
        # 專案根目錄 (假設腳本在 scripts 資料夾內)
        project_root = current_script_path.parent.parent 
        
        # 構建資料和模型的絕對路徑
        data_path = project_root / data_rel_path
        model_path = project_root / model_rel_path
        target_dir = project_root 

        self.device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
        print(f"Using Device: {self.device}")
        # -----------------------------------

        ## Claude!!
        # # 載入 spaCy
        # print("Loading spaCy...")

        # import os
        # print("Loading spaCy...")
        # try:
        #     self.nlp = spacy.load("en_core_web_md")
        #     print("✅ spaCy model loaded")
        # except OSError:
        #     print("Spacy model not found. Downloading...")
        #     import subprocess
        #     import sys
        #     subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
        #     self.nlp = spacy.load("en_core_web_md")
        #     print("✅ spaCy model downloaded and loaded")
        
        # 在 slangify_core.py 的 SlangifySystem.__init__ 內部

        print("Loading spaCy...")
        import subprocess
        import sys
        # -----------------------------

        try:
            self.nlp = spacy.load("en_core_web_md")
            print("✅ spaCy model loaded")
        except OSError:
            print("Spacy model not found. Downloading...")
            
            # 使用 --target 參數，將模型安裝到專案根目錄 (有寫入權限)
            try:
                subprocess.check_call([
                    sys.executable, 
                    "-m", "pip", "install", 
                    "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
                    f"--target={target_dir}", # 指定安裝到專案根目錄
                    "--no-deps" # 避免安裝多餘的依賴
                ])
            except Exception as e:
                print(f"❌ ERROR: Failed to install model to project dir. Error: {e}")
                raise

            # 必須將目標目錄加入 Python 搜尋路徑
            sys.path.append(str(target_dir))
            
            # 重新載入
            self.nlp = spacy.load("en_core_web_md")
            print("✅ spaCy model downloaded and loaded from project directory")
        
        
        # 載入資料
        print(f"Loading data: {data_path}")
        try:
            self.df = pd.read_csv(data_path)
            print(f"✅ Loaded {len(self.df)} slang entries")
        except FileNotFoundError:
            print(f"❌ ERROR: Data file not found at {data_path}")
            print("請檢查您的資料路徑是否正確。")
            raise # 重新拋出錯誤
        
        # # 建立 FAISS 索引
        # print("Building FAISS index...")
        # self.retriever = SentenceTransformer("all-MiniLM-L6-v2")
        # texts = self.df.apply(
        #     lambda r: f"{str(r['word'])} : {str(r['definition'])}", 
        #     axis=1
        # ).tolist()
        # embeddings = self.retriever.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        # self.index = faiss.IndexFlatIP(embeddings.shape[1])
        # self.index.add(embeddings)
        # print("✅ FAISS index built")
        
        

        
        
        # 建立 FAISS 索引
        print("Building FAISS index...")
        self.retriever = SentenceTransformer("all-MiniLM-L6-v2", device=self.device) 

        texts = self.df.apply(
            lambda r: f"{str(r['word'])} : {str(r['definition'])}", 
            axis=1
        ).tolist()

        # 關鍵優化：設置 num_workers
        # num_workers > 0 允許並行處理，但可能會增加記憶體使用量。
        # 通常設置為 4 或 8
        embeddings = self.retriever.encode(
            texts, 
            normalize_embeddings=True, 
            show_progress_bar=False,
            convert_to_tensor=True,
            batch_size=32, # 增大 batch size
            num_workers=4 # 啟用 4 個並行處理執行緒
        )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print("✅ FAISS index built")
        
        
        
        
        # 載入 BERT Classifier
        print(f"Loading BERT Classifier: {model_path}")
        self.device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2
        )
        
        try:
            # 載入模型權重
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            test_f1 = checkpoint.get('test_f1', checkpoint.get('val_f1', 0))
            print(f"✅ BERT loaded (Test F1: {test_f1:.4f}, Device: {self.device})")
        except FileNotFoundError:
            print(f"❌ ERROR: Model file not found at {model_path}")
            print("請檢查您的模型路徑是否正確。")
            raise
        except Exception as e:
             print(f"❌ ERROR: Loading BERT failed: {e}")
             raise
        
        print("\n✅ System ready!\n")
    
    # ============================================
    # Baseline Mode Functions
    # ============================================
    
    def extract_keywords_with_pos(self, sentence):
        """
        提取可替換的關鍵詞並標註詞性
        按優先級排序: ADJ > VERB > NOUN
        """
        doc = self.nlp(sentence)
        keywords = []
        
        for token in doc:
            lemma = token.lemma_.lower()
            text_lower = token.text.lower()
            
            # 黑名單過濾
            if lemma in SKIP_WORDS or text_lower in SKIP_WORDS:
                continue
            if len(token.text) <= 2 or token.is_stop:
                continue
            
            # 只保留 ADJ/VERB/NOUN
            if token.pos_ in {'ADJ', 'VERB', 'NOUN'}:
                # 處理片語動詞
                if token.pos_ == "VERB":
                    phrase_lemma = lemma
                    phrase_text = text_lower
                    for child in token.children:
                        if child.dep_ == "prt":
                            phrase_lemma = f"{lemma} {child.text.lower()}"
                            phrase_text = f"{text_lower} {child.text.lower()}"
                            break
                    keywords.append((phrase_lemma, phrase_text, token.pos_, token.i))
                else:
                    keywords.append((lemma, text_lower, token.pos_, token.i))
        
        # 按詞性優先級排序
        priority_map = {'ADJ': 0, 'VERB': 1, 'NOUN': 2}
        keywords.sort(key=lambda x: priority_map.get(x[2], 3))
        
        return keywords
    
    def retrieve_slang_faiss_by_keywords(self, sentence, keywords, 
                                        k_per_keyword=5, min_faiss_score=0.25):
        """
        為每個關鍵詞使用 FAISS 檢索候選 slang
        """
        all_results = []
        seen_words = set()
        
        for kw_lemma, kw_text, kw_pos, kw_idx in keywords:
            # FAISS 檢索
            q_emb = self.retriever.encode([kw_lemma], normalize_embeddings=True)
            scores, idxs = self.index.search(q_emb, k_per_keyword * 5)
            
            keyword_results = []
            for sc, ix in zip(scores[0], idxs[0]):
                if ix < 0 or ix >= len(self.df) or sc < min_faiss_score:
                    continue
                
                row = self.df.iloc[int(ix)]
                word = row["word"]
                word_lower = word.lower()
                definition = str(row["definition"]).lower()
                
                # 過濾
                if word_lower in seen_words or word_lower in NAME_BLACKLIST or word_lower in SKIP_WORDS:
                    continue
                if len(word) >= 15 or word.count(' ') >= 2:
                    continue
                if word[0].isupper() and not (word.isupper() and len(word) <= 5):
                    continue
                if '-' in word and ' ' in word:
                    continue
                if any(char in word for char in '[]{}()|<>@#$%^&*+=~`'):
                    continue
                if word_lower.endswith(('ster', 'sters', 'son', 'man', 'mann')):
                    if word_lower not in {'gangster', 'hipster', 'youngster', 'monster'}:
                        continue
                if len(word) > 5:
                    vowel_count = sum(1 for c in word_lower if c in 'aeiou')
                    if vowel_count / len(word) < 0.15:
                        continue
                if word_lower == kw_lemma or word_lower == kw_text:
                    continue
                if kw_lemma in word_lower or word_lower in kw_lemma:
                    continue
                
                # 匹配分數
                pattern = rf"^{re.escape(kw_lemma)}[\s,;]"
                if re.match(pattern, definition):
                    match_score = 1.0
                elif re.search(rf"\b{re.escape(kw_lemma)}\b", definition):
                    match_score = 0.6
                else:
                    match_score = 0.3
                
                keyword_results.append({
                    "score": float(sc),
                    "word": word,
                    "definition": row["definition"],
                    "example": row.get("example", ""),
                    "matched_keyword": kw_lemma,
                    "original_word": kw_text,
                    "original_pos": kw_pos,
                    "original_idx": kw_idx,
                    "match_score": match_score,
                })
                seen_words.add(word_lower)
                
                if len(keyword_results) >= k_per_keyword:
                    break
            
            all_results.extend(keyword_results)
        
        return all_results
    
    def score_candidates_with_classifier(self, sentence, candidates, alpha=0.35):
        """
        使用 BERT Classifier 評分並計算 Combined Score
        """
        if not candidates:
            return []
        
        # BERT 評分
        inputs = self.tokenizer(
            [sentence] * len(candidates),
            [f"{c['word']} : {c['definition']}" for c in candidates],
            padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        
        # 計算 Combined Score
        for i, c in enumerate(candidates):
            c["bert_score"] = float(probs[i])
            
            # 基礎分數
            base_score = alpha * c["score"] + (1 - alpha) * c["bert_score"]
            
            # 詞性加分
            pos_bonus = POS_PRIORITY.get(c.get("original_pos", ""), 0)
            
            # Definition 匹配加分
            match_bonus = 0.15 if c.get("match_score", 0) >= 1.0 else \
                         0.08 if c.get("match_score", 0) >= 0.6 else 0
            
            # 白名單加分
            c['is_popular'] = (c["word"].lower() in POPULAR_SLANG)
            popularity_bonus = 0.25 if c['is_popular'] else 0.05
            
            c["combined"] = base_score + pos_bonus + match_bonus + popularity_bonus
        
        # 排序
        candidates.sort(key=lambda x: x["combined"], reverse=True)
        return candidates
    
    def smart_replace(self, sentence, slang_word, original_word=None, original_idx=None):
        """智能替換"""
        doc = self.nlp(sentence)
        tokens = [token.text for token in doc]
        
        # 方法1: 使用 index
        if original_idx is not None and 0 <= original_idx < len(tokens):
            tokens[original_idx] = slang_word
            return self._rebuild_sentence(doc, tokens)
        
        # 方法2: 片語動詞
        if original_word and ' ' in original_word:
            pattern = r'\b' + r'\s+'.join(re.escape(p) for p in original_word.split()) + r'\b'
            result, count = re.subn(pattern, slang_word, sentence, flags=re.IGNORECASE)
            if count > 0:
                return result
        
        # 方法3: 單詞匹配
        if original_word:
            for i, token in enumerate(doc):
                if token.text.lower() == original_word or token.lemma_.lower() == original_word:
                    tokens[i] = slang_word
                    return self._rebuild_sentence(doc, tokens)
        
        return sentence
    
    def _rebuild_sentence(self, doc, tokens):
        """重建句子（保留標點符號間距）"""
        result = []
        for i, token in enumerate(doc):
            if i > 0 and not token.is_punct and not tokens[i-1].endswith("'"):
                result.append(" ")
            result.append(tokens[i])
        return "".join(result)
    
    # ============================================
    # Baseline Mode: 自動替換最佳候選
    # ============================================
    
    def slangify(self, sentence, k_per_keyword=5, conf_threshold=0.55, alpha=0.35):
        """
        Baseline Pipeline: 自動選擇最佳 slang 替換
        
        Parameters:
        -----------
        sentence : str
            輸入句子
        k_per_keyword : int
            每個關鍵詞取幾個候選
        conf_threshold : float
            信心門檻
        alpha : float
            FAISS vs BERT 權重
        
        Returns:
        --------
        tuple: (替換後的句子, 最佳候選詞資訊)
        """
        # Step 1: 提取關鍵詞
        keywords = self.extract_keywords_with_pos(sentence)
        if not keywords:
            return sentence, None
        
        # Step 2: FAISS 檢索
        candidates = self.retrieve_slang_faiss_by_keywords(
            sentence, keywords, k_per_keyword, min_faiss_score=0.25
        )
        if not candidates:
            return sentence, None
        
        # Step 3: BERT 評分
        candidates = self.score_candidates_with_classifier(sentence, candidates, alpha)
        
        # Step 4: 選擇最佳
        best = candidates[0]
        if best["combined"] < conf_threshold:
            return sentence, None
        
        # Step 5: 替換
        result = self.smart_replace(
            sentence, best["word"],
            best.get("original_word"),
            best.get("original_idx")
        )
        
        return result, best
    
    # ============================================
    # Interactive Mode: 多詞互動替換
    # ============================================
    
    def analyze_sentence(self, sentence):
        """
        分析句子，返回所有可替換的詞
        
        Returns:
        --------
        list of dict: [
            {
                'text': 'stylish',
                'lemma': 'stylish',
                'pos': 'ADJ',
                'index': 4,
                'replaceable': True
            },
            ...
        ]
        """
        doc = self.nlp(sentence)
        tokens = []
        
        for token in doc:
            lemma = token.lemma_.lower()
            text_lower = token.text.lower()
            
            # 判斷是否可替換
            replaceable = (
                token.pos_ in {'ADJ', 'ADV', 'VERB', 'NOUN'} and
                lemma not in SKIP_WORDS and
                text_lower not in SKIP_WORDS and
                len(token.text) > 2 and
                not token.is_stop
            )
            
            tokens.append({
                'text': token.text,
                'lemma': lemma,
                'pos': token.pos_,
                'index': token.i,
                'replaceable': replaceable
            })
        
        return tokens
    
    def get_suggestions(self, sentence, word_lemma, word_pos, word_index, top_k=5):
        """
        為單一詞獲取 slang 建議
        
        Parameters:
        -----------
        sentence : str
            完整句子
        word_lemma : str
            詞的 lemma
        word_pos : str
            詞性
        word_index : int
            詞在句子中的位置
        top_k : int
            返回前 k 個建議
        
        Returns:
        --------
        list of dict: [
            {
                'slang': 'fresh',
                'definition': 'new and stylish',
                'example': 'Those kicks are fresh.',
                'score': 0.92,
                'faiss_score': 0.48,
                'bert_score': 0.95,
                'is_popular': True,
                'preview': 'His outfit is really fresh'
            },
            ...
        ]
        """
        # FAISS 檢索
        q_emb = self.retriever.encode([word_lemma], normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb, top_k * 5)
        
        candidates = []
        seen_words = set()
        
        for sc, ix in zip(scores[0], idxs[0]):
            if ix < 0 or ix >= len(self.df) or sc < 0.25:
                continue
            
            row = self.df.iloc[int(ix)]
            word = row["word"]
            word_lower = word.lower()
            
            # 基本過濾
            if word_lower in seen_words or word_lower in SKIP_WORDS:
                continue
            if word_lower == word_lemma or word_lemma in word_lower:
                continue
            
            candidates.append({
                "word": word,
                "definition": str(row["definition"]),
                "example": str(row.get("example", "")),
                "faiss_score": float(sc)
            })
            seen_words.add(word_lower)
            
            if len(candidates) >= top_k * 2:
                break
        
        if not candidates:
            return []
        
        # BERT 評分
        inputs = self.tokenizer(
            [sentence] * len(candidates),
            [f"{c['word']} : {c['definition']}" for c in candidates],
            padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        
        # Combined Score
        for i, c in enumerate(candidates):
            c["bert_score"] = float(probs[i])
            base_score = 0.35 * c["faiss_score"] + 0.65 * c["bert_score"]
            pos_bonus = POS_PRIORITY.get(word_pos, 0)
            match_bonus = 0.10 if word_lemma in c["definition"].lower() else 0
            c['is_popular'] = (c["word"].lower() in POPULAR_SLANG)
            popularity_bonus = 0.25 if c['is_popular'] else 0.05
            c["combined"] = base_score + pos_bonus + match_bonus + popularity_bonus
            c["score"] = round(c["combined"], 2)
        
        # 排序
        candidates.sort(key=lambda x: x["combined"], reverse=True)
        
        # 生成預覽
        doc = self.nlp(sentence)
        for c in candidates[:top_k]:
            tokens = [t.text for t in doc]
            tokens[word_index] = c["word"]
            c['preview'] = ' '.join(tokens)
            
            # 清理
            c['slang'] = c.pop('word')
            del c['combined']
        
        return candidates[:top_k]
    
    def apply_replacements(self, sentence, replacements):
        """
        應用多個替換
        
        Parameters:
        -----------
        sentence : str
            原始句子
        replacements : dict
            {word_index: slang_word}
        
        Returns:
        --------
        str: 替換後的句子
        """
        doc = self.nlp(sentence)
        tokens = [t.text for t in doc]
        
        # 從大到小排序，避免 index 錯位
        for idx in sorted(replacements.keys(), reverse=True):
            if 0 <= idx < len(tokens):
                tokens[idx] = replacements[idx]
        
        return ' '.join(tokens)


# ============================================
# 測試函數
# ============================================

def test_system():
    """測試系統功能"""
    print("\n" + "="*60)
    print("Testing Slangify Core Module")
    print("="*60)
    
    # 這裡的 SlangifySystem() 會使用新的絕對路徑邏輯
    system = SlangifySystem()
    
    # Test 1: Baseline mode
    print("\n[Test 1] Baseline Mode (自動替換)")
    print("-"*60)
    
    test_sentences = [
        "He likes to show off his new car.",
        "That party was really amazing last night.",
        "She is upset because she lost the game.",
    ]
    
    for sent in test_sentences:
        result, best = system.slangify(sent)
        if best:
            print(f"\n原句: {sent}")
            print(f"結果: {result}")
            print(f"替換: {best['original_word']} → {best['word']} (分數: {best['combined']:.2f})")
        else:
            print(f"\n原句: {sent}")
            print("結果: 無替換")
    
    # Test 2: Interactive mode
    print("\n\n[Test 2] Interactive Mode (多詞替換)")
    print("-"*60)
    
    sentence = "His outfit is really stylish"
    print(f"\n句子: {sentence}")
    
    # 分析
    tokens = system.analyze_sentence(sentence)
    replaceable = [t for t in tokens if t['replaceable']]
    
    print(f"\n可替換詞: {[t['text'] for t in replaceable]}")
    
    # 為 "stylish" 獲取建議
    # 確保 replaceable 列表非空再訪問
    stylish_token = next((t for t in replaceable if t['lemma'] == 'stylish'), None)
    
    if stylish_token:
        suggestions = system.get_suggestions(
            sentence,
            stylish_token['lemma'],
            stylish_token['pos'],
            stylish_token['index'],
            top_k=3
        )
        
        print(f"\n'{stylish_token['text']}' 的建議:")
        for i, s in enumerate(suggestions, 1):
            star = "⭐" if s['is_popular'] else ""
            print(f"  {i}. {s['slang']:12} ({s['score']:.2f}) {star} - {s['definition'][:40]}...")
            print(f"     預覽: {s['preview']}")
        
        # 應用替換 (這裡假設 'really' 的 index 是 3)
        # 為了更安全，我們應該用 lemma 查找 index
        really_token = next((t for t in replaceable if t['lemma'] == 'really'), None)
        
        if really_token:
            replacements = {stylish_token['index']: 'chic', really_token['index']: 'fr'}  # stylish → chic, really → fr
            result = system.apply_replacements(sentence, replacements)
            
            print(f"\n多詞替換:")
            print(f"  原句: {sentence}")
            print(f"  結果: {result}")
        else:
            print("\n[SKIP] 找不到 'really' 詞，跳過多詞替換測試。")
            
    else:
        print("\n[SKIP] 找不到 'stylish' 詞，跳過 Interactive Mode 測試。")


    print("\n✅ All system initialization logic updated for robust path handling.")


if __name__ == "__main__":
    test_system()