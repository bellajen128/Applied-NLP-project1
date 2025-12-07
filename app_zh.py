"""
Slangify - Interactive Slang Generator
Streamlit Web Application
"""

import streamlit as st
import sys
sys.path.append('scripts')
from slangify_core import SlangifySystem

# ============================================
# 頁面配置
# ============================================

st.set_page_config(
    page_title="Slangify - Interactive Slang Generator",
    page_icon="🎨",
    layout="wide"
)

# ============================================
# 初始化系統（使用 session_state 避免重複載入）
# ============================================

@st.cache_resource
def load_system():
    """載入系統（只執行一次）"""
    return SlangifySystem(
        data_rel_path='data/slang_clean_final.csv',
        model_rel_path='models/best_slang_bert_classifier.pt',
        use_gpu=False  # Streamlit 通常在 CPU 環境
    )

# ============================================
# 主介面
# ============================================

def main():
    # 標題
    st.title("🎨 Slangify - Interactive Slang Generator")
    st.markdown("將你的句子轉換成潮流俚語！")
    
    # 載入系統
    with st.spinner("載入模型中..."):
        system = load_system()
    
    # 側邊欄：模式選擇
    st.sidebar.header("⚙️ 設定")
    mode = st.sidebar.radio(
        "選擇模式",
        ["🤖 自動模式 (Baseline)", "🎯 互動模式 (Interactive)"]
    )
    
    # ============================================
    # 模式 1: 自動模式
    # ============================================
    
    if mode == "🤖 自動模式 (Baseline)":
        st.header("🤖 自動模式")
        st.markdown("系統會自動選擇最適合的詞進行替換")
        
        # 輸入框
        sentence = st.text_input(
            "輸入句子：",
            placeholder="例如：He likes to show off his new car.",
            key="auto_input"
        )
        
        # 參數設定
        with st.sidebar.expander("🔧 進階參數"):
            conf_threshold = st.slider("信心門檻", 0.0, 1.0, 0.55, 0.05)
            alpha = st.slider("FAISS vs BERT 權重", 0.0, 1.0, 0.35, 0.05)
            k_per_keyword = st.slider("每個詞的候選數", 3, 10, 5)
        
        # 執行按鈕
        if st.button("🚀 Slangify!", key="auto_btn"):
            if sentence:
                with st.spinner("處理中..."):
                    result, best = system.slangify(
                        sentence,
                        k_per_keyword=k_per_keyword,
                        conf_threshold=conf_threshold,
                        alpha=alpha
                    )
                
                # 顯示結果
                if best:
                    st.success("✅ 替換成功！")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**原句：**")
                        st.info(sentence)
                    with col2:
                        st.markdown("**結果：**")
                        st.success(result)
                    
                    # 詳細資訊
                    with st.expander("📊 替換詳情"):
                        star = "⭐" if best.get('is_popular') else ""
                        st.markdown(f"**替換詞：** `{best['original_word']}` → `{best['word']}` {star}")
                        st.markdown(f"**定義：** {best['definition']}")
                        if best.get('example'):
                            st.markdown(f"**例句：** {best['example']}")
                        
                        st.markdown("**評分：**")
                        st.write(f"- Combined Score: `{best['bert_score']:.2f}`")
                        st.write(f"- FAISS Score: `{best['score']:.2f}`")
                        st.write(f"- BERT Score: `{best['bert_score']:.2f}`")
                else:
                    st.warning("❌ 找不到合適的替換詞（分數低於門檻）")
            else:
                st.error("請輸入句子！")
    
    # ============================================
    # 模式 2: 互動模式
    # ============================================
    
    else:  # Interactive Mode
        st.header("🎯 互動模式")
        st.markdown("選擇你想替換的詞，查看多個建議")
        
        # 輸入框
        sentence = st.text_input(
            "輸入句子：",
            placeholder="例如：His outfit is really stylish",
            key="interactive_input"
        )
        
        if sentence:
            # 分析句子
            with st.spinner("分析句子中..."):
                tokens = system.analyze_sentence(sentence)
            
            replaceable = [t for t in tokens if t['replaceable']]
            
            if not replaceable:
                st.warning("❌ 沒有可替換的詞")
            else:
                st.success(f"✅ 找到 {len(replaceable)} 個可替換的詞")
                
                # 顯示原句（標記可替換詞）
                st.markdown("**原句：**")
                highlighted = []
                for t in tokens:
                    if t['replaceable']:
                        highlighted.append(f"**[{t['text']}]**")
                    else:
                        highlighted.append(t['text'])
                st.info(" ".join(highlighted))
                
                # 使用 session_state 儲存選擇
                if 'selections' not in st.session_state:
                    st.session_state.selections = {}
                
                # 為每個可替換詞顯示建議
                st.markdown("---")
                st.markdown("### 💡 選擇要替換的詞")
                
                for word_info in replaceable:
                    with st.expander(f"🔹 {word_info['text']} ({word_info['pos']})"):
                        # 獲取建議按鈕
                        if st.button(f"獲取建議", key=f"get_{word_info['index']}"):
                            with st.spinner(f"搜尋 '{word_info['text']}' 的 slang..."):
                                suggestions = system.get_suggestions(
                                    sentence,
                                    word_info['lemma'],
                                    word_info['pos'],
                                    word_info['index'],
                                    top_k=5
                                )
                            
                            if suggestions:
                                st.session_state[f"suggestions_{word_info['index']}"] = suggestions
                            else:
                                st.warning("找不到合適的 slang")
                        
                        # 顯示建議
                        if f"suggestions_{word_info['index']}" in st.session_state:
                            suggestions = st.session_state[f"suggestions_{word_info['index']}"]
                            
                            st.markdown("**建議：**")
                            for i, s in enumerate(suggestions):
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    star = "⭐ " if s['is_popular'] else ""
                                    st.markdown(f"{star}**{s['slang']}** ({s['score']:.2f})")
                                    st.caption(s['definition'][:60])
                                
                                with col2:
                                    st.caption(f"FAISS: {s['faiss_score']:.2f}")
                                    st.caption(f"BERT: {s['bert_score']:.2f}")
                                
                                with col3:
                                    if st.button("選擇", key=f"select_{word_info['index']}_{i}"):
                                        st.session_state.selections[word_info['index']] = s['slang']
                                        st.success(f"✅ 已選擇: {s['slang']}")
                
                # 顯示當前選擇
                if word_info['index'] in st.session_state.selections:
                    current = st.session_state.selections[word_info['index']]
                    st.info(f"✅ 當前選擇: **{current}**")
                    if st.button("取消選擇", key=f"cancel_{word_info['index']}"):
                        del st.session_state.selections[word_info['index']]
                        st.rerun()
                
                st.markdown("---")
                
                # 預覽按鈕
                if st.session_state.selections:
                    preview = system.apply_replacements(sentence, st.session_state.selections)
                    st.markdown("### 🎬 預覽")
                    st.success(f"**{preview}**")
                    
                    if st.button("🔄 重置所有選擇"):
                        st.session_state.selections = {}
                        st.rerun()


# ============================================
# 側邊欄：系統資訊
# ============================================

def show_system_info():
    st.sidebar.markdown("---")
    st.sidebar.header("📊 系統資訊")
    st.sidebar.info(f"""
    **資料量:** 9,173 條 slang
    
    **BERT Classifier:**
    - Test F1: 0.9789
    - Val F1: 0.9795
    
    **模型架構:**
    - FAISS: all-MiniLM-L6-v2
    - Classifier: DistilBERT
    """)

show_system_info()


# ============================================
# 執行
# ============================================

if __name__ == "__main__":
    main()
