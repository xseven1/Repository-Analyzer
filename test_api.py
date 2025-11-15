from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if key is loaded
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {api_key[:10] if api_key else 'NOT FOUND'}...")

if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment!")
    print(f"Current directory: {os.getcwd()}")
    print(f".env file exists: {os.path.exists('.env')}")
    exit(1)

client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)