from typing import Union, List

from pydantic import BaseModel, Field


class TextContent(BaseModel):
    type: str = "text"
    text: str


class Url(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: Url


class FileContent(BaseModel):
    type: str = "file_url"
    file_url: Url


class RequestMessage(BaseModel):
    conversation_id: str
    model: str
    content: List[Union[TextContent, ImageContent, FileContent]]
    about_user_message: str = Field(default="")
    about_model_message: str = Field(default="")


class FirstResponseMessage(BaseModel):
    conversation_id: str
    thread_id: str
    thread_name: str
    new_thread: bool
    content: List[TextContent]
    video_urls: List[str]


class ResponseMessage(BaseModel):
    conversation_id: str
    thread_id: str
    content: List[TextContent]
    video_urls: List[str]
