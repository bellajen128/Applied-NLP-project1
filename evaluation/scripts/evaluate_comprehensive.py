"""
完整評估系統 - 使用 100 個真實測試句
包含 6 個指標
"""

import json
import numpy as np
import time
import torch
from collections import Counter
from slangify_core import SlangifySystem
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ComprehensiveEvaluator:
    """完整評估器"""
    
    def __init__(self):
        print("Initializing Comprehensive Evaluator...")
        self.system = SlangifySystem()
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 載入 GPT-2 用於 Perplexity
        print("Loading GPT-2 for perplexity calculation...")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_model.eval()
        
        print("✅ Evaluator ready\n")
    
    # ============================================
    # 指標 1.1: BERT Classifier Test F1 (已有)
    # ============================================
    
    def bert_test_f1(self):
        """直接從模型 checkpoint 讀取"""
        import torch
        checkpoint = torch.load(
            '../models/best_slang_bert_classifier.pt',
            map_location='cpu',
            weights_only=False
        )
        return checkpoint.get('test_f1', checkpoint.get('val_f1', 0))
    
    # ============================================
    # 指標 1.3: Match Score Distribution
    # ============================================
    
    def match_score_distribution(self, test_sentences):
        """
        檢查 FAISS 檢索的候選詞中，有多少真的匹配關鍵詞
        """
        print("\n[1.3] Match Score Distribution...")
        
        match_scores = []
        
        for sent_data in test_sentences:
            sentence = sent_data['sentence']
            
            # 提取關鍵詞
            keywords = self.system.extract_keywords_with_pos(sentence)
            if not keywords:
                continue
            
            # 只看第一個關鍵詞（最優先的）
            kw_lemma, kw_text, kw_pos, kw_idx = keywords[0]
            
            # 檢索候選
            candidates = self.system.retrieve_slang_faiss_by_keywords(
                sentence, [keywords[0]], k_per_keyword=5
            )
            
            # 收集 match_score
            for c in candidates:
                match_scores.append(c.get('match_score', 0))
        
        if match_scores:
            distribution = Counter(match_scores)
            print(f"  Match Score Distribution:")
            print(f"    1.0 (perfect match): {distribution.get(1.0, 0)} ({distribution.get(1.0, 0)/len(match_scores)*100:.1f}%)")
            print(f"    0.6 (contains keyword): {distribution.get(0.6, 0)} ({distribution.get(0.6, 0)/len(match_scores)*100:.1f}%)")
            print(f"    0.3 (semantic only): {distribution.get(0.3, 0)} ({distribution.get(0.3, 0)/len(match_scores)*100:.1f}%)")
            
            return {
                'distribution': dict(distribution),
                'mean': np.mean(match_scores),
                'perfect_match_rate': distribution.get(1.0, 0) / len(match_scores)
            }
        
        return None
    
    # ============================================
    # 指標 2.1: Replacement Rate (擴增測試)
    # ============================================
    
    def replacement_rate(self, test_sentences):
        """使用 100 個句子測試替換成功率"""
        print("\n[2.1] Replacement Rate (100 sentences)...")
        
        success_count = 0
        failures = []
        
        for sent_data in test_sentences:
            sentence = sent_data['sentence']
            result, best = self.system.slangify(sentence)
            
            if best:
                success_count += 1
            else:
                failures.append(sentence)
        
        rate = success_count / len(test_sentences)
        
        print(f"  ✅ Success: {success_count}/{len(test_sentences)} ({rate:.2%})")
        if failures:
            print(f"  Failed examples:")
            for f in failures[:3]:
                print(f"    - {f[:60]}...")
        
        return {
            'rate': rate,
            'success': success_count,
            'total': len(test_sentences),
            'failures': failures
        }
    
    # ============================================
    # 指標 2.3: Whitelist Coverage
    # ============================================
    
    def whitelist_coverage(self, test_sentences, k=3):
        """Top K 建議中白名單詞的比例"""
        print("\n[2.3] Whitelist Coverage (Top-3)...")
        
        popular_count = 0
        total_count = 0
        
        for sent_data in test_sentences:
            sentence = sent_data['sentence']
            
            # 提取第一個關鍵詞
            keywords = self.system.extract_keywords_with_pos(sentence)
            if not keywords:
                continue
            
            kw_lemma, kw_text, kw_pos, kw_idx = keywords[0]
            
            # 獲取建議
            suggestions = self.system.get_suggestions(
                sentence, kw_lemma, kw_pos, kw_idx, top_k=k
            )
            
            for s in suggestions:
                total_count += 1
                if s['is_popular']:
                    popular_count += 1
        
        rate = popular_count / total_count if total_count > 0 else 0
        
        print(f"  ✅ Whitelist coverage: {rate:.2%} ({popular_count}/{total_count})")
        
        return {
            'rate': rate,
            'whitelist_count': popular_count,
            'total_suggestions': total_count
        }
    
    # ============================================
    # 指標 3.1: Semantic Preservation
    # ============================================
    
    def semantic_preservation(self, test_sentences):
        """替換前後的語義相似度"""
        print("\n[3.1] Semantic Preservation...")
        
        similarities = []
        
        for sent_data in test_sentences:
            original = sent_data['sentence']
            result, best = self.system.slangify(original)
            
            if best:
                emb1 = self.semantic_model.encode([original])
                emb2 = self.semantic_model.encode([result])
                sim = util.cos_sim(emb1, emb2)[0][0].item()
                similarities.append(sim)
        
        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            
            print(f"  ✅ Mean: {mean_sim:.3f}, Std: {std_sim:.3f}")
            print(f"  Range: [{min(similarities):.3f}, {max(similarities):.3f}]")
            
            return {
                'mean': mean_sim,
                'std': std_sim,
                'min': min(similarities),
                'max': max(similarities)
            }
        
        return None
    
    # ============================================
    # 指標 4.1: Perplexity
    # ============================================
    
    def calculate_perplexity(self, test_sentences):
        """
        使用 GPT-2 計算句子困惑度
        越低越自然
        """
        print("\n[4.1] Perplexity (Fluency)...")
        
        original_ppls = []
        result_ppls = []
        
        for sent_data in test_sentences:
            original = sent_data['sentence']
            result, best = self.system.slangify(original)
            
            # 計算原句 perplexity
            orig_ppl = self._compute_perplexity(original)
            original_ppls.append(orig_ppl)
            
            # 計算替換後 perplexity
            if best:
                res_ppl = self._compute_perplexity(result)
                result_ppls.append(res_ppl)
        
        if original_ppls and result_ppls:
            print(f"  Original sentences:")
            print(f"    Mean PPL: {np.mean(original_ppls):.2f}")
            print(f"  Replaced sentences:")
            print(f"    Mean PPL: {np.mean(result_ppls):.2f}")
            print(f"  Δ PPL: {np.mean(result_ppls) - np.mean(original_ppls):+.2f}")
            
            return {
                'original_mean': np.mean(original_ppls),
                'result_mean': np.mean(result_ppls),
                'delta': np.mean(result_ppls) - np.mean(original_ppls)
            }
        
        return None
    
    def _compute_perplexity(self, sentence):
        """計算單一句子的 perplexity"""
        encodings = self.gpt2_tokenizer(sentence, return_tensors='pt')
        input_ids = encodings.input_ids
        
        with torch.no_grad():
            outputs = self.gpt2_model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    
    # ============================================
    # 完整評估
    # ============================================
    
    def run_evaluation(self, test_file='test_sentences_from_examples.json'):
        """執行完整評估"""
        
        # 載入測試句子
        print(f"Loading test sentences from: {test_file}")
        with open(test_file, 'r') as f:
            test_sentences = json.load(f)
        
        print(f"✅ Loaded {len(test_sentences)} test sentences\n")
        
        print("="*60)
        print("Running Comprehensive Evaluation")
        print("="*60)
        
        results = {}
        
        # 指標 1.1: BERT Test F1
        print("\n[1.1] BERT Classifier Test F1...")
        bert_f1 = self.bert_test_f1()
        results['bert_test_f1'] = bert_f1
        print(f"  ✅ Test F1: {bert_f1:.4f}")
        
        # 指標 1.3: Match Score Distribution
        match_dist = self.match_score_distribution(test_sentences)
        results['match_score_distribution'] = match_dist
        
        # 指標 2.1: Replacement Rate
        rep_rate = self.replacement_rate(test_sentences)
        results['replacement_rate'] = rep_rate
        
        # 指標 2.3: Whitelist Coverage
        whitelist_cov = self.whitelist_coverage(test_sentences, k=3)
        results['whitelist_coverage'] = whitelist_cov
        
        # 指標 3.1: Semantic Preservation
        sem_pres = self.semantic_preservation(test_sentences)
        results['semantic_preservation'] = sem_pres
        
        # 指標 4.1: Perplexity
        perplexity = self.calculate_perplexity(test_sentences)
        results['perplexity'] = perplexity
        
        # 保存結果
        print("\n" + "="*60)
        print("Saving Results")
        print("="*60)
        
        with open('comprehensive_evaluation_results.json', 'w') as f:
            # 移除 details 和 failures（太長）
            summary = {
                'bert_test_f1': results['bert_test_f1'],
                'match_score_distribution': {
                    'mean': results['match_score_distribution']['mean'],
                    'perfect_match_rate': results['match_score_distribution']['perfect_match_rate']
                } if results['match_score_distribution'] else None,
                'replacement_rate': results['replacement_rate']['rate'],
                'whitelist_coverage': results['whitelist_coverage']['rate'],
                'semantic_preservation': results['semantic_preservation']['mean'] if results['semantic_preservation'] else None,
                'perplexity_delta': results['perplexity']['delta'] if results['perplexity'] else None
            }
            json.dump(summary, f, indent=2)
        
        print("✅ Saved to: comprehensive_evaluation_results.json")
        
        # 打印摘要
        print("\n" + "="*60)
        print("EVALUATION SUMMARY (100 Test Sentences)")
        print("="*60)
        print(f"1.1 BERT Test F1:           {results['bert_test_f1']:.4f}")
        if results['match_score_distribution']:
            print(f"1.3 Perfect Match Rate:     {results['match_score_distribution']['perfect_match_rate']:.2%}")
        print(f"2.1 Replacement Rate:       {results['replacement_rate']['rate']:.2%}")
        print(f"2.3 Whitelist Coverage:     {results['whitelist_coverage']['rate']:.2%}")
        if results['semantic_preservation']:
            print(f"3.1 Semantic Preservation:  {results['semantic_preservation']['mean']:.3f}")
        if results['perplexity']:
            print(f"4.1 Perplexity Delta:       {results['perplexity']['delta']:+.2f}")
        print("="*60)
        
        return results


if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_evaluation('test_sentences_from_examples.json')
    
    print("\n🎉 Evaluation complete!")
