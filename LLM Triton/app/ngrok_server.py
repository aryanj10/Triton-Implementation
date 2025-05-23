import os
from pyngrok import ngrok
import uvicorn
from dotenv import load_dotenv

load_dotenv()

def run_app():
    ngrok_token = os.getenv("NGROK_TOKEN")
    if not ngrok_token:
        raise ValueError("NGROK_TOKEN environment variable is not set")

    ngrok.set_auth_token(ngrok_token)

    # Tunnel to port 8081 (or whatever port your FastAPI runs on)
    public_url = ngrok.connect(8081).public_url
    if public_url.startswith('http://'):
        public_url = 'https://' + public_url[7:]

    print(f"🔗 Public NGROK URL: {public_url}")

    # Optionally use this URL in app logic
    os.environ["REDIRECT_URI"] = f"{public_url}/callback"

    # Run your FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=True)

if __name__ == "__main__":
    run_app()
