import google.generativeai as genai
import traceback

# API key provided by the user
GEMINI_API_KEY = "AIzaSyA0rSuOuFw_hCsX5WKJ8akisuCPC8P7TAM"

print("Testing Gemini API connection...")

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
    
    # Test prompt
    prompt = "Analyze the following news text: 'Scientists discover a new planet in our solar system that was previously hidden.' Is this fake news? Respond with 'Fake' or 'Real'."
    
    print("Sending request to Gemini API...")
    response = model.generate_content(prompt)
    
    print("Response received:")
    print(response.text)
    print("\nAPI test successful!")
    
except Exception as e:
    print("Error occurred:")
    print(str(e))
    print("\nTraceback:")
    traceback.print_exc()
    print("\nAPI test failed!")