import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_confidence_gauge(confidence, prediction):
    """
    Create a gauge chart showing the confidence level of the prediction
    
    Args:
        confidence (float): Confidence percentage
        prediction (int): 0 for real, 1 for fake
    """
    # Determine gauge color based on prediction and confidence
    if prediction == 1:  # Fake news
        color = 'rgba(255, 99, 71, 0.8)'  # Red for fake
    else:  # Real news
        color = 'rgba(60, 179, 113, 0.8)'  # Green for real
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(255, 255, 255, 0.1)'},
                {'range': [30, 70], 'color': 'rgba(255, 255, 255, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(255, 255, 255, 0.5)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(feature_importance):
    """
    Create a horizontal bar chart for feature importance
    
    Args:
        feature_importance (list): List of (word, importance) tuples
    """
    if not feature_importance:
        st.warning("No significant features found for explanation.")
        return
    
    # Extract words and importance scores
    words = [item[0] for item in feature_importance]
    scores = [item[1] for item in feature_importance]
    
    # Determine colors based on scores
    colors = ['#FF6B6B' if score > 0 else '#4ECDC4' for score in scores]
    
    # Create labels for the legend
    fake_indicator = px.scatter(x=[0], y=[0], color_discrete_sequence=['#FF6B6B'])
    real_indicator = px.scatter(x=[0], y=[0], color_discrete_sequence=['#4ECDC4'])
    
    # Create the bar chart
    fig = px.bar(
        x=scores,
        y=words,
        orientation='h',
        labels={'x': 'Importance Score', 'y': 'Feature'},
        color_discrete_sequence=['#bebada'],
        text=[f"{abs(score):.3f}" for score in scores]
    )
    
    # Customize the chart
    fig.update_traces(
        marker_color=colors,
        textposition='outside',
        textfont=dict(size=12)
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Influence on Classification",
        yaxis_title=None,
        legend_title="Influence Direction",
        hovermode="y"
    )
    
    # Add a line at x=0
    fig.add_shape(
        type="line",
        x0=0, y0=-0.5,
        x1=0, y1=len(words)-0.5,
        line=dict(color="black", width=1, dash="dot")
    )
    
    # Add annotation to explain colors
    fig.add_annotation(
        x=max(scores) if max(scores) > 0 else min(scores),
        y=len(words) + 0.5,
        text="Red bars indicate features suggesting fake news, green bars suggest real news",
        showarrow=False,
        font=dict(size=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_word_cloud(text, max_words=100):
    """
    Generate a word cloud visualization from the text
    
    Args:
        text (str): Preprocessed text
        max_words (int): Maximum number of words to include
    """
    try:
        from wordcloud import WordCloud
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            max_words=max_words, 
            background_color='white',
            colormap='viridis'
        ).generate(text)
        
        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
        
    except ImportError:
        # If wordcloud is not available, show a message
        st.info("WordCloud visualization is not available.")
