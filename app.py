"""
Slangify - Interactive Slang Generator
Streamlit Web Application
"""

import streamlit as st
import pandas as pd # Need pandas for dataframe in system info
import sys
# Set path to allow importing slangify_core.py from the scripts directory
sys.path.append('scripts') 

try:
    from slangify_core import SlangifySystem
except ImportError:
    st.error("Error: Could not import SlangifySystem. Please ensure 'slangify_core.py' is in the 'scripts' directory and all dependencies are installed.")
    st.stop()
    
# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Slangify - Interactive Slang Generator",
    page_icon="🤙🏿",
    layout="wide"
)

# ============================================
# System Initialization (Caching)
# ============================================

@st.cache_resource
def load_system():
    """Loads the Slangify System (runs only once)."""
    # NOTE: Using corrected parameter names: data_rel_path and model_rel_path
    return SlangifySystem(
        data_rel_path='data/slang_clean_final.csv',
        model_rel_path='models/best_slang_bert_classifier.pt',
        use_gpu=False  # Streamlit typically runs in CPU environment
    )

# ============================================
# Main Interface
# ============================================

def main():
    # Title
    st.title("Slangify - Interactive Slang Generator")
    st.markdown("Convert your sentences into modern slang expressions.")
    
    # Load System
    with st.spinner("Loading models..."):
        try:
            system = load_system()
        except Exception as e:
            st.error(f"Failed to load Slangify System. Check model/data paths and dependencies. Error: {e}")
            return
    
    # Sidebar: Mode Selection
    st.sidebar.header("Settings")
    mode = st.sidebar.radio(
        "Select Mode",
        ["Automatic Mode (Baseline)", "Interactive Mode"]
    )
    
    # --- Mode 1: Automatic Mode ---
    
    if mode == "Automatic Mode (Baseline)":
        st.header("Automatic Mode")
        st.markdown("The system automatically selects the most appropriate word for replacement.")
        
        # Input
        sentence = st.text_input(
            "Enter Sentence:",
            placeholder="E.g., He likes to show off his new car.",
            key="auto_input"
        )
        
        # Parameters
        with st.sidebar.expander("Advanced Parameters"):
            conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.55, 0.05)
            alpha = st.slider("FAISS vs BERT Weight (Alpha)", 0.0, 1.0, 0.35, 0.05)
            k_per_keyword = st.slider("Candidates per Keyword (K)", 3, 10, 5)
        
        # Execution Button
        if st.button("Slangify!", key="auto_btn"):
            if sentence:
                with st.spinner("Processing..."):
                    result, best = system.slangify(
                        sentence,
                        k_per_keyword=k_per_keyword,
                        conf_threshold=conf_threshold,
                        alpha=alpha
                    )
                
                # Display Result
                if best:
                    st.success("Replacement Successful!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Sentence:**")
                        st.info(sentence)
                    with col2:
                        st.markdown("**Result:**")
                        st.success(result)
                    
                    # Detailed Information
                    with st.expander("Replacement Details"):
                        star = "[Popular]" if best.get('is_popular') else ""
                        st.markdown(f"**Replaced:** `{best['original_word']}` -> `{best['word']}` {star}")
                        st.markdown(f"**Definition:** {best['definition']}")
                        if best.get('example'):
                            st.markdown(f"**Example Usage:** {best['example']}")
                        
                        st.markdown("**Scoring:**")
                        # Display scores using st.dataframe for better formatting
                        st.dataframe(pd.DataFrame({
                            'Score Type': ['Combined', 'FAISS (Semantic)', 'BERT (Contextual)'],
                            'Value': [best.get('combined', 0), best.get('score', 0), best.get('bert_score', 0)]
                        }).set_index('Score Type'))
                        
                else:
                    st.warning("No suitable replacement found (score below threshold).")
                    st.info(sentence)
            else:
                st.error("Please enter a sentence!")
    
    # --- Mode 2: Interactive Mode ---
    
    else:  # Interactive Mode
        st.header("Interactive Mode")
        st.markdown("Select a word to replace and view multiple suggestions.")
        
        # Input
        sentence = st.text_input(
            "Enter Sentence:",
            placeholder="E.g., His outfit is really stylish",
            key="interactive_input"
        )
        
        if sentence:
            # Analyze Sentence
            with st.spinner("Analyzing sentence..."):
                tokens = system.analyze_sentence(sentence)
            
            replaceable = [t for t in tokens if t['replaceable']]
            
            if not replaceable:
                st.warning("No replaceable words found.")
            else:
                st.success(f"Found {len(replaceable)} replaceable words.")
                
                # Display Original Sentence (highlighting replaceable words)
                st.markdown("**Original Sentence (Replaceable words in brackets):**")
                highlighted = []
                for t in tokens:
                    if t['replaceable']:
                        highlighted.append(f"**[{t['text']}]**")
                    else:
                        highlighted.append(t['text'])
                st.info(" ".join(highlighted))
                
                # Use session_state to store selection
                if 'selections' not in st.session_state:
                    st.session_state.selections = {}
                
                # Display suggestions for each replaceable word
                st.markdown("---")
                st.subheader("Select Slang Replacements")
                
                for word_info in replaceable:
                    with st.expander(f"Word: {word_info['text']} ({word_info['pos']})"):
                        # Get Suggestions Button (always retrieve suggestions when expander is opened)
                        
                        # Use a key to store suggestions per word
                        suggestions_key = f"suggestions_{word_info['index']}"
                        
                        # --- Logic to retrieve suggestions (Runs on button click or first render) ---
                        if st.button(f"Get Suggestions", key=f"get_{word_info['index']}"):
                            with st.spinner(f"Searching for '{word_info['text']}' slang..."):
                                suggestions = system.get_suggestions(
                                    sentence,
                                    word_info['lemma'],
                                    word_info['pos'],
                                    word_info['index'],
                                    top_k=5
                                )
                            
                            if suggestions:
                                st.session_state[suggestions_key] = suggestions
                            else:
                                st.warning("No suitable slang found.")
                            # Rerun immediately after fetching to display results
                            st.rerun() 
                        
                        # --- Display Suggestions ---
                        if suggestions_key in st.session_state:
                            suggestions = st.session_state[suggestions_key]
                            
                            st.markdown("**Suggestions:**")
                            for i, s in enumerate(suggestions):
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    star = "[Popular] " if s['is_popular'] else ""
                                    st.markdown(f"{star}**{s['slang']}** (Score: {s['score']:.2f})")
                                    st.caption(s['definition'][:60] + "...")
                                
                                with col2:
                                    st.caption(f"FAISS: {s['faiss_score']:.2f}")
                                    st.caption(f"BERT: {s['bert_score']:.2f}")
                                
                                with col3:
                                    if st.button("Select", key=f"select_{word_info['index']}_{i}"):
                                        st.session_state.selections[word_info['index']] = s['slang']
                                        st.rerun() # Rerun to update selection status and preview
                        
                        # Display current selection status inside the expander
                        if word_info['index'] in st.session_state.selections:
                            current = st.session_state.selections[word_info['index']]
                            st.info(f"Selected: **{current}**")
                            if st.button("Deselect", key=f"cancel_{word_info['index']}"):
                                del st.session_state.selections[word_info['index']]
                                st.rerun()
                
                st.markdown("---")
                
                # Preview Section
                if st.session_state.selections:
                    preview = system.apply_replacements(sentence, st.session_state.selections)
                    st.markdown("### Preview")
                    st.success(f"**{preview}**")
                    
                    if st.button("Reset All Selections"):
                        st.session_state.selections = {}
                        st.rerun()
                else:
                    st.info("No replacements selected yet. Choose a slang word above.")


# ============================================
# Sidebar: System Information
# ============================================

def show_system_info():
    st.sidebar.markdown("---")
    st.sidebar.header("System Metrics")
    st.sidebar.info(f"""
    **Data Size:** 9,173 Slang Entries
    
    **BERT Classifier:**
    - Test F1: 0.9789
    - Val F1: 0.9795
    
    **Model Architecture:**
    - Retrieval: all-MiniLM-L6-v2 (FAISS)
    - Classifier: DistilBERT
    """)

show_system_info()


# ============================================
# Execution
# ============================================

if __name__ == "__main__":
    main()