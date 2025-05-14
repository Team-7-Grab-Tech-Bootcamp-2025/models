import json
from typing import List
import PIL
import cv2
import numpy as np
import torch
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print
from transformers.image_utils import load_image
from rich.console import Console
from copy import deepcopy

console = Console()


def init_model(name, device):
    """
    Initialize the model and processor.
    :param name: Model name
    :param device: Device to use (e.g., "cuda" or "cpu")
    :return: Tuple of processor and model
    """
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer, model


async def read_image_file(file) -> np.ndarray:
    """
    Reads an image from a file object and returns it as a NumPy array.

    Args:
        file: A file-like object containing an image.

    Returns:
        A NumPy array representing the image.
    """

    contents = await file.read()

    rgb_image = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
    return rgb_image


class ChatClient:
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-Instruct-500M", device=None):
        """
        Initialize the ChatClient with a model name and device.
        :param model_name: Name of the model to use
        :param device: Device to use (e.g., "cuda" or "cpu")
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = init_model(model_name, self.device)
        self.sessions_path = "./sessions.json"
        self.system_prompt = (
            "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực trả lời các câu hỏi về ăn uống. Hãy luôn trả lời một cách hữu ích nhất có thể. Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết. "
            "Nếu câu hỏi không liên quan đến ăn uống, hãy từ chối trả lời và yêu cầu người dùng hỏi về ăn uống."
            "Khi được hỏi bạn có thể làm gì, hãy liệt kê các khả năng của bạn. Điều này bao gồm việc cung cấp thông tin về các nhà hàng, món ăn, đánh giá của người dùng"
        )
        self.template = """
Hãy trả lời câu hỏi dựa trên ngữ cảnh:
### Ngữ cảnh :
{context}

### Câu hỏi :
{question}

### Trả lời :"""

    def chat(self, session_id, user_text_input, image_pb=[]):
        """
        Process the chat messages and return the generated response.
        :param session_id: ID of the chat session
        :param messages: List of messages in the chat
        :return: Generated response text
        """
        sessions = self.load_sessions()
        messages = sessions.get(session_id, [])
        if session_id not in sessions:
            messages.append({"role": "system", "content": self.system_prompt})
        try:
            context = ""
            messages.append(
                {
                    "role": "user",
                    "content": self.template.format(
                        context=context, question=user_text_input
                    ),
                }
            )
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=2048,
                temperature=0.1,
                # top_p=0.95,
                # top_k=40,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            assistant_msg = response
            # Append the assistant's response to the messages
            messages.append({"role": "assistant", "content": assistant_msg})
            # Save the updated session
            sessions[session_id] = messages

            self.save_sessions(sessions)
            # Return the generated response
            return assistant_msg

        except Exception as e:
            console.print_exception()
            return ""

    def load_sessions(self):
        """
        Load sessions from a JSON file.
        :return: Dictionary of sessions
        """
        try:
            with open(self.sessions_path, "r") as f:
                sessions = json.load(f)
        except FileNotFoundError:
            sessions = {}
        except json.JSONDecodeError:
            sessions = {}

        return sessions

    def save_sessions(self, sessions):
        """
        Save sessions to a JSON file.
        """
        try:
            with open(self.sessions_path, "w") as f:
                json.dump(sessions, f, indent=4)
        except ValueError as e:
            console.print_exception()
            raise ValueError(f"Failed to save sessions: {e}")
