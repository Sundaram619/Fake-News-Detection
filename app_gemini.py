"""
Fake News Detection Application using Streamlit and Google's Gemini API
"""

import streamlit as st
import os
import time
import re
import pandas as pd
import numpy as np

# Import the Gemini classifier
from gemini_classifier import analyze_with_gemini

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector with Gemini AI",
    page_icon="🔍",
    layout="wide"
)

def main():
    """Main application function"""
    # Display header
    st.title("Fake News Detection System powered by Gemini AI")
    st.markdown("""
    This application uses Google's Gemini AI to analyze and detect potentially fake news articles. 
    Enter an article text or upload a text file for analysis.
    """)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Enter Text", "Upload File"])
    
    with tab1:
        # Text input area
        article_text = st.text_area("Paste your article text here:", height=250)
        analyze_button = st.button("Analyze Text", key="analyze_text")
        
        if analyze_button and article_text:
            analyze_and_display_results(article_text)
        elif analyze_button and not article_text:
            st.warning("Please enter some text to analyze.")
    
    with tab2:
        # File upload option
        uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
        
        if uploaded_file is not None:
            try:
                # Read the file as text
                article_text = uploaded_file.read().decode("utf-8")
                st.text_area("File Content Preview:", article_text[:500] + "..." if len(article_text) > 500 else article_text, height=150)
                analyze_file_button = st.button("Analyze File", key="analyze_file")
                
                if analyze_file_button:
                    analyze_and_display_results(article_text)
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # About section
    with st.expander("About this Fake News Detector"):
        st.markdown("""
        ### How It Works
        
        This application uses Google's Gemini AI for advanced text analysis:
        
        1. **Article Analysis**: Your article is sent to Gemini for detailed language processing.
        2. **Pattern Recognition**: Gemini identifies linguistic and contextual patterns associated with fake news.
        3. **Classification**: Based on its analysis, Gemini classifies the content as fake or reliable news.
        4. **Detailed Explanation**: The AI provides a full explanation of its reasoning and key indicators.
        
        ### Privacy Note
        
        Text you enter is sent to Google's Gemini API for analysis. Please don't enter sensitive personal information.
        
        ### Best Practices
        
        Always verify information from multiple reliable sources before forming conclusions, even with AI assistance.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2024 Fake News Detection System | Powered by Google Gemini")

def analyze_and_display_results(text):
    """Analyze the text with Gemini and display the results"""
    if not text or len(text.strip()) < 20:
        st.warning("Please provide a longer article for accurate analysis (at least 20 characters).")
        return
    
    # Show processing indicator
    with st.spinner('Analyzing text with Gemini AI...'):
        try:
            # Make prediction using Gemini
            is_fake, confidence, explanation, features = analyze_with_gemini(text)
            
            # Scale confidence to percentage
            confidence_pct = confidence * 100
            
            # Display results
            st.markdown("---")
            st.header("Analysis Results")
            
            # Set up columns for results display
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Show classification result with clear labeling
                if is_fake:  # Fake news
                    result_color = "red"
                    result_text = "FAKE NEWS"
                    st.markdown(f"""
                    <div style='background-color:rgba(255,0,0,0.1); padding:15px; border-radius:10px; border:3px solid red;'>
                        <h1 style='color:{result_color}; text-align:center; margin:0;'>{result_text}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Real news
                    result_color = "green"
                    result_text = "RELIABLE NEWS"
                    st.markdown(f"""
                    <div style='background-color:rgba(0,128,0,0.1); padding:15px; border-radius:10px; border:3px solid green;'>
                        <h1 style='color:{result_color}; text-align:center; margin:0;'>{result_text}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show confidence meter
                st.subheader("Confidence Score")
                st.progress(confidence)
                st.markdown(f"<h3 style='text-align:center; color:{result_color};'>{confidence_pct:.1f}%</h3>", unsafe_allow_html=True)
                
                # Show Gemini's explanation
                st.subheader("Gemini's Analysis")
                st.write(explanation)
                
                # Add source attribution
                st.caption("Analysis provided by Google's Gemini AI")
                
            with col2:
                # Show factor analysis
                st.subheader("Key Factors in Analysis")
                create_factor_visualization(features, is_fake)
                
                # Show text analysis
                st.subheader("Text Analysis")
                st.text_area("Article excerpt:", text[:400] + "..." if len(text) > 400 else text, height=200)
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            # Show the full error details for debugging
            import traceback
            st.error(traceback.format_exc())
            
            # Suggest ways to fix the issue
            st.warning("""
            **Troubleshooting Tips:**
            - Check your internet connection
            - Verify the API key is correct
            - Try with a different piece of text
            - The API may have rate limits or other restrictions
            """)

def create_factor_visualization(features, is_fake):
    """Create a visualization of key factors"""
    # Create a dataframe for the features
    feature_df = pd.DataFrame(features, columns=['Feature', 'Importance'])
    feature_df = feature_df.sort_values('Importance', ascending=False).head(5)
    
    # Determine colors based on classification
    colors = ['#ff6b6b'] * len(feature_df) if is_fake else ['#6bff6b'] * len(feature_df)
    
    # Display as a horizontal bar chart
    st.bar_chart(feature_df.set_index('Feature'), use_container_width=True)
    
    # Also display as a bullet list with colored bullets
    for feature, importance in features[:5]:  # Show top 5 features
        if is_fake:
            bullet_color = "red"
        else:
            bullet_color = "green"
        
        st.markdown(f"<span style='color:{bullet_color};'>●</span> <b>{feature}</b> (impact: {importance:.2f})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()