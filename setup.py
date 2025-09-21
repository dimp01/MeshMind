import os
from pathlib import Path

def main():
    print("=== MeshMind Setup ===\n")
    
    # Prompt for Gemini API Key
    gemini_key = input("Enter your GEMINI_API_KEY: ").strip()
    while not gemini_key:
        gemini_key = input("GEMINI_API_KEY cannot be empty. Enter your GEMINI_API_KEY: ").strip()
    
    # Prompt for ngrok auth token
    ngrok_token = input("Enter your NGROK_AUTH_TOKEN: ").strip()
    while not ngrok_token:
        ngrok_token = input("NGROK_AUTH_TOKEN cannot be empty. Enter your NGROK_AUTH_TOKEN: ").strip()
    
    # Write to .env file
    env_path = Path(".env")
    with env_path.open("w") as f:
        f.write(f"GEMINI_API_KEY={gemini_key}\n")
        f.write(f"NGROK_AUTH_TOKEN={ngrok_token}\n")
    
    print("\n✅ .env file created successfully!")
    print(f"Location: {env_path.resolve()}")
    print("You can now run main.py and your credentials will be used automatically.\n")

if __name__ == "__main__":
    main()
