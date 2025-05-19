import google.generativeai as genai
import json
import traceback

# API key provided by the user
GEMINI_API_KEY = "AIzaSyA0rSuOuFw_hCsX5WKJ8akisuCPC8P7TAM"

print("Testing Gemini API with JSON response format...")

try:
    # Configure the API key
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Set up the model
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    
    # Create the model instance
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config
    )
    
    # Test text
    test_text = "Scientists discover a new planet in our solar system that was previously hidden"
    
    # Test prompt
    prompt = f"""
    Analyze the following news text to determine if it's fake news or reliable news. 
    
    Examples of fake news:
    - Claims about extraordinary discoveries without scientific consensus
    - Claims about new planets in our solar system
    - Conspiracy theories about government coverups
    - Miracle cures or treatments that sound too good to be true
    - Shocking claims without credible sources
    
    Respond with a JSON object containing:
    1. "is_fake": true if fake news, false if reliable news
    2. "confidence": a number between 0 and 1 indicating your confidence
    3. "explanation": detailed explanation of why you classified it as fake or reliable
    4. "key_features": an array of objects, each with "feature" and "importance" (between 0 and 1) properties
    
    News text to analyze:
    "{test_text}"
    
    Respond with ONLY the JSON object, no additional text.
    """
    
    print("Sending request to Gemini API...")
    response = model.generate_content(prompt)
    
    print("Response received. Raw response:")
    print(response.text)
    
    # Clean the response if needed
    result = response.text
    if "```" in result:
        # Extract content between triple backticks
        import re
        json_match = re.search(r'```(?:json)?(.*?)```', result, re.DOTALL)
        if json_match:
            result = json_match.group(1).strip()
            print("\nExtracted JSON from markdown code block:")
            print(result)
    
    # Try to parse the JSON
    try:
        analysis = json.loads(result)
        print("\nSuccessfully parsed JSON response! Structure:")
        print(json.dumps(analysis, indent=2))
        
        # Check key properties
        print("\nExtracting key properties:")
        is_fake = analysis.get("is_fake", None)
        confidence = analysis.get("confidence", None)
        explanation = analysis.get("explanation", None)
        key_features = analysis.get("key_features", None)
        
        print(f"is_fake: {is_fake}")
        print(f"confidence: {confidence}")
        print(f"explanation: {explanation[:100]}..." if explanation else "None")
        print(f"key_features: {key_features}")
        
    except json.JSONDecodeError as json_err:
        print(f"\nJSON parsing error: {str(json_err)}")
        print("Failed to parse the response as valid JSON.")
    
except Exception as e:
    print("Error occurred:")
    print(str(e))
    print("\nTraceback:")
    traceback.print_exc()