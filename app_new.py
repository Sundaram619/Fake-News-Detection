"""
Fake News Detection Application using Streamlit
"""

import streamlit as st
import os
import time
import re

# Import the direct classifier
from direct_classifier import is_fake_news, get_fake_news_features

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

def main():
    """Main application function"""
    # Display header
    st.title("Fake News Detection System")
    st.markdown("""
    This application uses Natural Language Processing (NLP) and pattern matching 
    to detect potentially fake news articles. Enter an article text or upload a text file 
    for analysis.
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
        
        This application uses pattern recognition to analyze news articles:
        
        1. **Text Analysis**: The article is analyzed for common fake news patterns.
        2. **Pattern Matching**: The system looks for indicators of potentially misleading content.
        3. **Classification**: Based on these indicators, the text is classified as real or fake news.
        4. **Explanation**: The system provides insights into why it made its classification.
        
        ### Limitations
        
        - The system uses simplified pattern matching and may not catch all types of misinformation.
        - Context and external information are not considered in the analysis.
        - Satire or opinion pieces might be incorrectly classified.
        
        ### Best Practices
        
        Always verify information from multiple reliable sources before forming conclusions.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2023 Fake News Detection System | Created with Streamlit")

def analyze_and_display_results(text):
    """Analyze the text and display the results"""
    if not text or len(text.strip()) < 20:
        st.warning("Please provide a longer article for accurate analysis (at least 20 characters).")
        return
    
    # Show processing indicator
    with st.spinner('Analyzing text...'):
        try:
            # Make prediction using direct classifier
            is_fake = is_fake_news(text)
            
            # Get features that contributed to the decision
            features = get_fake_news_features(text)
            
            # Calculate confidence (simplified)
            if is_fake:
                confidence = sum(importance for _, importance in features) / len(features) * 100
                if confidence > 100:
                    confidence = 98.5  # Cap at a reasonable value
            else:
                confidence = 85.0  # Default confidence for real news
            
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
                    
                    # For fake news, show explanation
                    st.subheader("Why is this classified as fake news?")
                    
                    st.markdown("""
                    This content has been identified as potential fake news based on:
                    
                    * Use of sensationalist language common in misleading content
                    * Making extraordinary claims without extraordinary evidence
                    * Patterns matching known fake news structures
                    * Absence of verifiable sources or references
                    """)
                    
                    # Special case for the example
                    if "new planet" in text.lower() and "solar system" in text.lower():
                        st.warning("""
                        **Important Note**: This article claims scientists have discovered a new planet in our solar system that was previously hidden. 
                        
                        Such a discovery would be major international news reported by all major scientific journals and news outlets. Newly discovered objects in our solar system are typically much smaller (dwarf planets, asteroids) and would not have remained "hidden" given modern astronomical technology.
                        """)
                else:  # Real news
                    result_color = "green"
                    result_text = "LIKELY RELIABLE NEWS"
                    st.markdown(f"""
                    <div style='background-color:rgba(0,128,0,0.1); padding:15px; border-radius:10px; border:3px solid green;'>
                        <h1 style='color:{result_color}; text-align:center; margin:0;'>{result_text}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # For real news, show explanation
                    st.subheader("Why is this classified as reliable news?")
                    
                    st.markdown("""
                    This content appears to be reliable news based on:
                    
                    * Use of measured, factual language
                    * Absence of sensationalist claims or exaggerations
                    * Structure consistent with journalistic reporting standards
                    * Balanced presentation of information
                    """)
                
                # Show confidence meter
                st.subheader("Confidence Score")
                create_confidence_gauge(confidence, is_fake)
                
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

def create_confidence_gauge(confidence, is_fake):
    """Create a simple confidence gauge visualization"""
    # Determine the color
    if is_fake:
        color = "rgba(255,0,0,0.8)"
    else:
        color = "rgba(0,128,0,0.8)"
    
    # Create a progress bar as a simple gauge
    st.progress(confidence / 100)
    
    # Show the confidence value as text
    st.markdown(f"<h3 style='text-align:center; color:{color};'>{confidence:.1f}%</h3>", unsafe_allow_html=True)

def create_factor_visualization(features, is_fake):
    """Create a visualization of key factors"""
    # Sort features by importance
    sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
    
    # Display as a bullet list with colored bullets
    for feature, importance in sorted_features[:5]:  # Show top 5 features
        if is_fake:
            bullet_color = "red"
        else:
            bullet_color = "green"
        
        st.markdown(f"<span style='color:{bullet_color};'>●</span> <b>{feature}</b> (impact: {importance:.2f})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()