import streamlit as st

"""
Statistics Panel Component for Swiss German ASR Dashboard
Provides clear explanations of key ASR metrics used throughout the dashboard.
"""

def render_metrics_definitions():
    """
    Render an expandable section with metrics definitions, formulas, and explanations.
    """
    with st.expander("📊 Metrics Definitions & Formulas"):
        st.markdown("### Word Error Rate (WER)")
        st.latex(r"WER = \frac{S + D + I}{N} \times 100\%")
        st.markdown("""
        **Plain Language:** WER measures how many words were inserted (I), deleted (D), 
        or substituted (S) compared to the total number of words in the reference (N). 
        Lower is better - 0% means perfect transcription.
        """)
        st.markdown("[📖 Learn more about WER](https://en.wikipedia.org/wiki/Word_error_rate)")
        
        st.markdown("---")
        
        st.markdown("### Character Error Rate (CER)")
        st.latex(r"CER = \frac{S + D + I}{N} \times 100\%")
        st.markdown("""
        **Plain Language:** Similar to WER, but calculated at the character level instead 
        of word level. Useful for languages with complex word structures or when evaluating 
        character-by-character accuracy. Lower is better.
        """)
        st.markdown("[📖 Learn more about CER](https://www.isca-speech.org/archive/interspeech_2005/morris05_interspeech.html)")
        
        st.markdown("---")
        
        st.markdown("### BLEU Score")
        st.latex(r"BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)")
        st.markdown("""
        **Plain Language:** BLEU measures how many word sequences (n-grams) from the 
        hypothesis appear in the reference. Originally designed for machine translation, 
        it's also useful for ASR. Higher is better - 100 means perfect match.
        """)
        st.markdown("[📖 BLEU Paper (Papineni et al., 2002)](https://aclanthology.org/P02-1040.pdf)")
        
        st.markdown("---")
        
        st.info("""
        **💡 Tip:** For Swiss German ASR, WER and CER are particularly important due to 
        dialect variations and non-standard spellings. BLEU can help evaluate semantic similarity.
        """)