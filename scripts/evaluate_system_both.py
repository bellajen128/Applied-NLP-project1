"""
Slangify 系統自動評估
包含 5 個自動指標 - 支援雙測試集評估
"""

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from slangify_core import SlangifySystem
from sentence_transformers import SentenceTransformer, util

# ============================================
# 輔助函式：載入外部 JSON 測試用例
# ============================================

def load_test_cases(file_path):
    """從 JSON 檔案載入測試用例"""
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    data_path = project_root / file_path
    
    print(f"\nLoading test cases from: {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        return test_cases
    except Exception as e:
        print(f"❌ ERROR: Failed to load JSON file. Error: {e}")
        return [] 

# ============================================
# 輔助函式：硬編碼 10 句測試用例
# ============================================

def create_default_test_cases():
    """建立 10 個硬編碼的測試用例 (用於對比)"""
    return [
        {'sentence': "He likes to show off his new car.", 'target_word': 'new', 'target_lemma': 'new', 'target_pos': 'ADJ', 'target_index': 6},
        {'sentence': "That party was really amazing last night.", 'target_word': 'amazing', 'target_lemma': 'amazing', 'target_pos': 'ADJ', 'target_index': 4},
        {'sentence': "She is upset because she lost the game.", 'target_word': 'upset', 'target_lemma': 'upset', 'target_pos': 'ADJ', 'target_index': 2},
        {'sentence': "I need to leave soon.", 'target_word': 'leave', 'target_lemma': 'leave', 'target_pos': 'VERB', 'target_index': 3},
        {'sentence': "He is acting very suspicious today.", 'target_word': 'suspicious', 'target_lemma': 'suspicious', 'target_pos': 'ADJ', 'target_index': 4},
        {'sentence': "This food tastes terrible.", 'target_word': 'terrible', 'target_lemma': 'terrible', 'target_pos': 'ADJ', 'target_index': 3},
        {'sentence': "She's so smart and clever.", 'target_word': 'smart', 'target_lemma': 'smart', 'target_pos': 'ADJ', 'target_index': 2},
        {'sentence': "Let's go home now.", 'target_word': 'go', 'target_lemma': 'go', 'target_pos': 'VERB', 'target_index': 1},
        {'sentence': "That movie was incredible.", 'target_word': 'incredible', 'target_lemma': 'incredible', 'target_pos': 'ADJ', 'target_index': 3},
        {'sentence': "He's feeling sad today.", 'target_word': 'sad', 'target_lemma': 'sad', 'target_pos': 'ADJ', 'target_index': 2},
    ]


# ============================================
# 評估器類別 (核心邏輯不變)
# ============================================

class ComprehensiveEvaluator:
    """評估系統"""
    
    def __init__(self):
        print("Initializing Evaluator...")
        self.system = SlangifySystem() 
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2') 
        print("✅ Evaluator ready\n")
    
    # --------------------------------------------
    # 指標 1-5 函式保持不變
    # --------------------------------------------
    # 指標 1: Replacement Rate 
    def replacement_rate(self, test_cases):
        success_count = 0
        failures = []
        for case in test_cases:
            result, best = self.system.slangify(case['sentence'])
            if best: success_count += 1
            else: failures.append(case['sentence'])
        rate = success_count / len(test_cases) if test_cases else 0
        return {'rate': rate, 'success': success_count, 'total': len(test_cases), 'failures': failures}
    
    # 指標 2: Semantic Preservation
    def semantic_preservation(self, test_cases):
        similarities = []
        for case in test_cases:
            original = case['sentence']
            result, best = self.system.slangify(original)
            if best:
                emb1 = self.semantic_model.encode([original])
                emb2 = self.semantic_model.encode([result])
                sim = util.cos_sim(emb1, emb2)[0][0].item()
                similarities.append(sim)
        if similarities:
            return {'mean': np.mean(similarities), 'std': np.std(similarities), 
                    'min': np.min(similarities), 'max': np.max(similarities)}
        return None
        
    # 指標 3: Popularity Rate
    def popularity_rate(self, test_cases, k=3):
        popular_count = 0
        total_count = 0
        for case in test_cases:
            if 'target_lemma' not in case or 'target_pos' not in case or 'target_index' not in case: continue
            suggestions = self.system.get_suggestions(
                case['sentence'], case['target_lemma'], case['target_pos'], case['target_index'], top_k=k)
            for s in suggestions:
                total_count += 1
                if s.get('is_popular'): popular_count += 1
        rate = popular_count / total_count if total_count > 0 else 0
        return {'rate': rate, 'popular_count': popular_count, 'total_count': total_count}

    # 指標 4: Average BERT Confidence
    def average_bert_confidence(self, test_cases):
        bert_scores = []
        for case in test_cases:
            result, best = self.system.slangify(case['sentence'])
            if best: bert_scores.append(best['bert_score'])
        if bert_scores:
            return {'mean': np.mean(bert_scores), 'std': np.std(bert_scores), 
                    'min': np.min(bert_scores), 'max': np.max(bert_scores)}
        return None

    # 指標 5: Inference Time
    def inference_time(self, test_cases):
        times = []
        for case in test_cases:
            start = time.time()
            self.system.slangify(case['sentence'])
            times.append(time.time() - start)
        if times:
            return {'mean': np.mean(times), 'std': np.std(times)}
        return None
    # --------------------------------------------
    
    def comprehensive_evaluation(self, test_cases, title):
        """執行所有指標評估，並打印結果"""
        if not test_cases: return {}
        
        print("\n" + "="*60)
        print(f"[{title}] Comprehensive Evaluation ({len(test_cases)} cases)")
        print("="*60)
        
        results = {}
        
        results['replacement_rate'] = rep_rate = self.replacement_rate(test_cases)
        results['semantic_preservation'] = sem_pres = self.semantic_preservation(test_cases)
        results['popularity_rate'] = pop_rate = self.popularity_rate(test_cases, k=3)
        results['bert_confidence'] = bert_conf = self.average_bert_confidence(test_cases)
        results['inference_time'] = inf_time = self.inference_time(test_cases)
        
        # 打印摘要
        print(f"Replacement Rate:     {rep_rate['rate']:.2%}")
        print(f"Semantic Similarity:  {sem_pres['mean']:.3f}" if sem_pres else "N/A")
        print(f"Popularity Rate:      {pop_rate['rate']:.2%}")
        print(f"BERT Confidence:      {bert_conf['mean']:.3f}" if bert_conf else "N/A")
        print(f"Avg Inference Time:   {inf_time['mean']:.3f}s" if inf_time else "N/A")
        
        return results

    def run_all_evaluations(self, json_file_name):
        """執行雙測試集的評估流程"""
        
        # 1. 載入評估器 (只執行一次)
        
        # 2. 評估 1：硬編碼 10 句測試集
        default_cases = create_default_test_cases()
        results_10 = self.comprehensive_evaluation(default_cases, "DEFAULT 10-CASE SET")
        
        # 3. 評估 2：100 句 JSON 測試集
        test_cases_100 = load_test_cases(json_file_name)
        results_100 = self.comprehensive_evaluation(test_cases_100, "CORPUS 100-CASE SET")

        # 4. 準備最終儲存 (合併結果)
        
        final_summary = {
            'DEFAULT_10_CASES': {
                'replacement_rate': results_10['replacement_rate']['rate'],
                'semantic_preservation_mean': results_10['semantic_preservation']['mean'] if results_10.get('semantic_preservation') else None,
                'popularity_rate': results_10['popularity_rate']['rate'],
                'bert_confidence_mean': results_10['bert_confidence']['mean'] if results_10.get('bert_confidence') else None,
                'inference_time_mean': results_10['inference_time']['mean']['mean'] if results_10.get('inference_time') else None,
            },
            'CORPUS_100_CASES': {
                'replacement_rate': results_100['replacement_rate']['rate'],
                'semantic_preservation_mean': results_100['semantic_preservation']['mean'] if results_100.get('semantic_preservation') else None,
                'popularity_rate': results_100['popularity_rate']['rate'],
                'bert_confidence_mean': results_100['bert_confidence']['mean'] if results_100.get('bert_confidence') else None,
                'inference_time_mean': results_100['inference_time']['mean']['mean'] if results_100.get('inference_time') else None,
            }
        }
        
        output_file_name = 'evaluation_dual_summary.json'
        with open(output_file_name, 'w') as f:
            json.dump(final_summary, f, indent=2)
            
        print("\n" + "="*60)
        print(f"✅ Final Summary saved to: {output_file_name}")
        print("="*60)
        return final_summary


# ============================================
# 主程式區塊
# ============================================

if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    # 運行雙測試集評估
    results = evaluator.run_all_evaluations('test_sentences_corpus_100.json') 
    
    print("\n🎉 Evaluation complete!")