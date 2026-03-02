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

        st.markdown("### chrF Score")
        st.latex(r"chrF = F_{\beta}(char\text{-}precision,\ char\text{-}recall)")
        st.markdown("""
        **Plain Language:** chrF measures character-level n-gram overlap between the hypothesis 
        and reference. More robust than word-level metrics for morphologically rich languages 
        like German, and less sensitive to minor spelling variations. Higher is better — 100 
        means perfect character-level match.
        """)
        st.markdown("[📖 chrF Paper (Popović, 2015)](https://aclanthology.org/W15-3049.pdf)")

        st.markdown("---")

        st.markdown("### Semantic Distance (SemDist)")
        st.latex(r"SemDist = 1 - \frac{\mathbf{e}_{ref} \cdot \mathbf{e}_{hyp}}{\|\mathbf{e}_{ref}\| \cdot \|\mathbf{e}_{hyp}\|}")
        st.markdown("""
        **Plain Language:** SemDist uses multilingual sentence embeddings to measure how 
        different the *meaning* of two sentences is. A score of 0 means identical meaning, 
        1 means completely unrelated. Unlike WER or BLEU, it tolerates paraphrases and 
        dialect-level rewording that preserve meaning. Lower is better.
        """)

        st.markdown("---")

        st.info("""
        **💡 Tip:** For Swiss German ASR, WER and CER are particularly important due to 
        dialect variations and non-standard spellings. chrF is more robust than BLEU for 
        German morphology. SemDist captures meaning preservation beyond surface-level overlap.
        """)