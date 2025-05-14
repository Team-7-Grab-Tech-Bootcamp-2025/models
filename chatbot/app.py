import os
import uuid
import ngrok
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Form
from fastapi.responses import RedirectResponse, StreamingResponse
import uvicorn
from fastapi.responses import ORJSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from rich.console import Console
from pathlib import Path
from module.chat_model import ChatClient

console = Console()

IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

app = FastAPI(title="Chat API", default_response_class=ORJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
chat_client = ChatClient(
    model_name="AITeamVN/Vi-Qwen2-1.5B-RAG",
)


@app.get("/", include_in_schema=False)
async def index():
    """
    This endpoint redirects the root URL to the API documentation page.

    Returns:
        RedirectResponse: A redirection to the API documentation page.
    """
    return RedirectResponse(url="/docs")


@app.post("/chat", tags=["Chat"])
async def chat(
    session_id: str = Form(...),
    user_text_input: str = Form(...),
    image_pb: Optional[List[UploadFile]] = None,
):
    """
    Process the chat messages and return the generated response.

    Args:

        session_id (str): Unique identifier for the chat session.
        user_text_input (str): User's text input for the chat.
        image_pb (Optional[List[UploadFile]]): List of uploaded image files.

    Returns:

        dict: A dictionary containing the status, messages, and response ID.
        Example:
            {
                "status": "success",
                "messages": "Hello, how can I help you?",
                "res_id": [] # List of relevant IDs
            }

    """
    try:
        # Process uploaded image files
        image_contents = []
        if image_pb is None:
            image_pb = []
        for img in image_pb:
            contents = await img.read()

            # Generate unique filename and check for collisions
            file_extension = Path(img.filename).suffix
            safe_path = ""
            max_attempts = 5  # Limit retries to prevent infinite loops

            for attempt in range(max_attempts):
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                safe_path = f"{IMG_DIR}/{unique_filename}"

                # Check if file already exists (handles the extremely rare UUID collision)
                if not os.path.exists(safe_path):
                    break

                # If we've reached max attempts, use timestamp + random component
                if attempt == max_attempts - 1:
                    import time
                    import random

                    unique_filename = f"{int(time.time())}_{random.randint(10000, 99999)}{file_extension}"
                    safe_path = f"{IMG_DIR}/{unique_filename}"

            # Save to disk with unique name
            with open(safe_path, "wb") as f:
                f.write(contents)

            # Add the unique path to image_contents
            image_contents.append(safe_path)

        # Pass the image data to the chat client
        response = chat_client.chat(session_id, user_text_input, image_contents)
        res_ids = []
        if user_text_input == "test_res_id":
            res_ids = ["B_1", "B_2", "B_3"]

        data = {"status": "success", "messages": response, "res_id": res_ids}

        return data
    except ValueError as e:
        console.print_exception()
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    TOKEN = "2XsWpDq5NFe8KIDz6wphz8ODahb_52KixEd4odr9LRRCtm7rV"
    PORT = 31456
    # Set ngrok domain
    # Start ngrok tunnel
    listener = ngrok.forward(
        PORT,
        domain="precious-needed-bug.ngrok-free.app",
        authtoken=TOKEN,
    )

    print(f"Ngrok tunnel started at: {listener.url()}")
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=PORT)
