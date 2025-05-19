import random

def get_explanation(prediction, probability, feature_importance):
    """
    Generate an explanation for the classification result
    
    Args:
        prediction (int): 0 for real, 1 for fake
        probability (float): Prediction probability
        feature_importance (list): List of (word, importance) tuples
        
    Returns:
        str: Explanation text
    """
    confidence_level = probability * 100
    
    if prediction == 1:  # Fake news
        explanation = generate_fake_news_explanation(confidence_level, feature_importance)
    else:  # Real news
        explanation = generate_real_news_explanation(confidence_level, feature_importance)
    
    return explanation

def generate_fake_news_explanation(confidence_level, feature_importance):
    """Generate explanation for fake news prediction"""
    
    # Base explanation
    explanation = f"This article has been classified as **potentially fake news** with a confidence level of {confidence_level:.1f}%.\n\n"
    
    # Different explanations based on confidence level
    if confidence_level > 90:
        explanation += "The text contains **strong indicators** of misleading information commonly found in fake news articles. "
    elif confidence_level > 70:
        explanation += "The text shows **several characteristics** typical of misleading content. "
    else:
        explanation += "The text has **some elements** that are associated with potentially misleading content. "
    
    # Add information about the features
    if feature_importance:
        # Extract specific known fake news indicators
        sensationalist_markers = [word for word, importance in feature_importance 
                                if 'sensationalist' in word or 'clickbait' in word or 
                                   'conspiracy' in word or 'emphasis' in word or
                                   'many_exclamations' in word or 'many_questions' in word]
        
        # Extract other positive importance features (those contributing to fake classification)
        fake_indicators = [word for word, importance in feature_importance 
                          if importance > 0 and word not in sensationalist_markers]
        
        explanation += "\n\n**Key indicators of potential misinformation:**\n"
        
        # Mention sensationalist language if detected
        if sensationalist_markers:
            formatted_markers = []
            for marker in sensationalist_markers:
                if 'sensationalist_headline' in marker:
                    formatted_markers.append("sensationalist headlines (e.g., 'BREAKING', 'SHOCKING')")
                elif 'excessive_emphasis' in marker:
                    formatted_markers.append("excessive emphasis (e.g., ALL CAPS, multiple exclamation marks)")
                elif 'conspiracy_language' in marker:
                    formatted_markers.append("conspiracy-related terms (e.g., 'secret', 'they don't want you to know')")
                elif 'clickbait_words' in marker:
                    formatted_markers.append("clickbait phrases (e.g., 'miracle', 'you won't believe')")
                elif 'many_exclamations' in marker:
                    formatted_markers.append("excessive use of exclamation marks")
                elif 'many_questions' in marker:
                    formatted_markers.append("excessive use of question marks")
            
            if formatted_markers:
                explanation += f"- Use of **sensationalist language techniques**: {', '.join(formatted_markers[:3])}\n"
        else:
            explanation += "- Use of sensationalist language or exaggerated claims\n"
            
        # Mention specific indicative words
        if fake_indicators:
            explanation += f"- Presence of **terms often found in misleading content**: {', '.join(fake_indicators[:3])}\n"
        
        # Additional context
        explanation += "- Narrative patterns and language structure consistent with known fake news articles\n"
        
        # Add accuracy strength statement based on confidence
        if confidence_level > 85:
            explanation += "- **Strong match** with linguistic patterns found in known misleading articles\n"
    
    # Add cautionary note
    explanation += "\n\n**Remember:** This is an automated analysis and may not be 100% accurate. Always verify information from multiple reliable sources."
    
    return explanation

def generate_real_news_explanation(confidence_level, feature_importance):
    """Generate explanation for real news prediction"""
    
    # Base explanation
    explanation = f"This article has been classified as **likely reliable news** with a confidence level of {confidence_level:.1f}%.\n\n"
    
    # Different explanations based on confidence level
    if confidence_level > 90:
        explanation += "The text contains **strong indicators** of factual reporting commonly found in legitimate news sources. "
    elif confidence_level > 70:
        explanation += "The text shows **several characteristics** typical of reliable news content. "
    else:
        explanation += "The text has **some elements** that are associated with reliable reporting, but caution is still advised. "
    
    # Add information about the features
    if feature_importance:
        # Check for absence of sensationalist language
        has_sensationalist = any('sensationalist' in word or 'clickbait' in word or 
                               'conspiracy' in word or 'emphasis' in word or
                               'many_exclamations' in word or 'many_questions' in word 
                               for word, _ in feature_importance)
        
        # Extract negative importance features (those contributing to real classification)
        real_indicators = [word for word, importance in feature_importance if importance < 0]
        
        explanation += "\n\n**Key indicators of reliable information:**\n"
        
        # Mention lack of sensationalism if applicable
        if not has_sensationalist:
            explanation += "- **Absence of sensationalist language** typical of fake news (no excessive capitalization, minimal use of emotionally charged terms)\n"
        
        # Mention factual reporting style
        explanation += "- Balanced and measured tone typical of professional journalism\n"
        
        # Mention specific indicative words if available
        if real_indicators:
            filtered_indicators = [term for term in real_indicators if len(term) > 3 and not term.startswith('domain_')]
            if filtered_indicators:
                explanation += f"- Presence of **terminology common in factual reporting**: {', '.join(filtered_indicators[:3])}\n"
        
        # Add contextual indicators
        explanation += "- Structure and presentation consistent with established news reporting standards\n"
        
        # Add confidence-based statement
        if confidence_level > 85:
            explanation += "- **Strong match** with linguistic patterns found in verified news sources\n"
            
        # Add source-related information if domain was detected
        domain_indicators = [word.replace('domain_', '') for word, _ in feature_importance if word.startswith('domain_')]
        if domain_indicators:
            explanation += "- Content structure similar to articles from established news outlets\n"
    
    # Add cautionary note with more nuance
    explanation += "\n\n**Remember:** This is an automated analysis based on text patterns. Even reliable-seeming content should be evaluated critically and verified with multiple sources."
    
    return explanation

def format_article_preview(text, max_length=200):
    """
    Format article text for preview display
    
    Args:
        text (str): Article text
        max_length (int): Maximum length for preview
        
    Returns:
        str: Formatted preview text
    """
    if len(text) <= max_length:
        return text
    
    # Truncate to max_length and find the last space
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + "..."
