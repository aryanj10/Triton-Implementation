import os
from pyngrok import ngrok
import uvicorn
import json
from dotenv import load_dotenv

load_dotenv()
def run_app():
    # Configure ngrok
    ngrok_token = os.getenv("NGROK_TOKEN")
    if not ngrok_token:
        raise ValueError("NGROK_TOKEN environment variable is not set")

    ngrok.set_auth_token(ngrok_token)

    # Start ngrok tunnel
    public_url = ngrok.connect(8081).public_url

    # Convert http to https
    if public_url.startswith('http://'):
        public_url = 'https://' + public_url[7:]

    print(f"NGrok URL: {public_url}")

    # Set the REDIRECT_URI environment variable
    os.environ["REDIRECT_URI"] = f"{public_url}/callback"

    # Start the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=True)

if __name__ == "__main__":
    run_app()