import streamlit as st

"""
Terminology Panel Component for Swiss German ASR Dashboard

Provides clear explanations of key ASR terminology used throughout the dashboard.
"""

def render_terminology_definitions():
    """
    Render an expandable section with terminology definitions and explanations.
    """
    with st.expander("📖 Terminology Explained"):
        st.markdown("### Reference")
        st.markdown("""
        **Plain Language:** The reference is the correct, ground-truth transcription of 
        the audio. It's what the speaker actually said, manually transcribed by humans. It can also be read speech or scripted text. 
        We use this as the "gold standard" to compare against the model's predictions.
        """)
        st.info("💡 **Example:** If someone says *'Grüezi mitenand'*, the reference text is exactly that.")
        
        st.markdown("---")
        
        st.markdown("### Hypothesis")
        st.markdown("""
        **Plain Language:** The hypothesis (alternatively called "prediction") is what the ASR model transcribed 
        from the audio. It's the model's "best guess" at what was said. We compare this 
        against the reference to calculate error rates and evaluate model performance.
        """)
        st.info("💡 **Example:** The model might transcribe *'Grüezi mitenand'* as *'Grüezi miteinander'* - that's the hypothesis.")
        
        st.markdown("---")
        
        st.markdown("### Word-Level Alignment")
        st.markdown("""
        **Plain Language:** Word-level alignment shows exactly how the reference and 
        hypothesis match up word-by-word. It identifies which words were:
        - **Correct (✓):** The word matches exactly in both reference and hypothesis.
        - **Substituted (✗):** A different word was used in the hypothesis than in the reference.
        - **Deleted (-):** A word present in the reference is missing from the hypothesis.
        - **Inserted (+):** An extra word appears in the hypothesis that wasn't in the reference.
        
        This helps us understand *where* and *how* the model makes mistakes.
        """)

        st.markdown("**💡 Example Alignment:**")
        st.code("""
        REF:  Grüezi  mitenand  wie  gaht's
        HYP:  Grüezi  miteinander  ---  gaht's
        TYPE:    ✓         ✗         -      ✓
        """, language=None)

        st.caption("""
        Here we see a substitution (*miteinander* vs *mitenand*) 
        and a deletion (missing *wie*).
        """)
        
        st.markdown("---")
        
        st.success("""
        **🎯 Why This Matters:** Understanding these terms helps you interpret the error 
        analysis and identify patterns in model performance across different Swiss German dialects.
        """)