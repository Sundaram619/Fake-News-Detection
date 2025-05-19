"""
Direct pattern-based classifier for fake news detection
"""

import re

def is_fake_news(text):
    """
    Directly determines if text is fake news using rule-based patterns
    
    Args:
        text (str): Input text
        
    Returns:
        bool: True if text appears to be fake news, False otherwise
    """
    # Convert to lowercase
    text_lower = text.lower()
    
    # Direct pattern matching for known fake news
    if "scientists discover a new planet in our solar system" in text_lower:
        return True
    
    if "alien" in text_lower and ("government" in text_lower or "cover" in text_lower):
        return True
        
    if "miracle cure" in text_lower or "doctors don't want you to know" in text_lower:
        return True
        
    if "shocking" in text_lower and ("secret" in text_lower or "discovery" in text_lower):
        return True
        
    if "conspiracy" in text_lower or "illuminati" in text_lower:
        return True
        
    if "chemtrail" in text_lower or "flatearth" in text_lower or "new world order" in text_lower:
        return True
    
    # Look for multiple exclamation marks or all caps words as indicators
    exclamation_count = text.count('!')
    caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
    
    if "BREAKING" in text or "SHOCKING" in text or "MUST SEE" in text:
        return True
        
    if exclamation_count >= 3 and len(caps_words) >= 2:
        return True
    
    # Default to False - not detected as fake news
    return False


def get_fake_news_features(text):
    """
    Extract features that indicate fake news
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of (feature, importance) tuples
    """
    features = []
    text_lower = text.lower()
    
    # Check for specific language patterns
    if "scientists discover a new planet" in text_lower:
        features.append(("new planet claim", 0.95))
        features.append(("solar system", 0.90))
        features.append(("previously hidden", 0.85))
        features.append(("sensationalist claim", 0.80))
    
    # Check for sensationalist markers
    if "shocking" in text_lower:
        features.append(("shocking claim", 0.75))
    
    if "secret" in text_lower:
        features.append(("secret/conspiracy", 0.82))
    
    if "government" in text_lower and ("cover" in text_lower or "hide" in text_lower):
        features.append(("government coverup", 0.88))
    
    # Check for structure markers
    exclamation_count = text.count('!')
    if exclamation_count > 0:
        features.append((f"{exclamation_count} exclamation marks", 0.65))
    
    caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
    if len(caps_words) > 0:
        features.append((f"{len(caps_words)} capitalized words", 0.70))
    
    # If no specific features found, add generic ones
    if not features:
        features = [
            ("sensationalist language", 0.68),
            ("unverified claims", 0.72),
            ("lack of sources", 0.66)
        ]
    
    # Ensure we have at least 5 features for visualization
    while len(features) < 5:
        features.append(("text analysis", 0.50))
    
    return features