import datetime
import re
from typing import Annotated, Union

from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.constant import AzureOpenAICFG, CoreCFG
from src.config.prompt import (
    CHATGPT_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    customize_prompt,
)
from src.config.model import TextContent, ImageContent, FileContent
from src.utils.logger import logger
from src.module.graph.tools import tools
from src.module.storage.blob import download_file


class State(TypedDict):
    thread_name: str
    conversation_id: str
    model: str
    content: list[Union[TextContent, ImageContent, FileContent]]
    messages: Annotated[
        list[Union[TextContent, ImageContent, FileContent]], add_messages
    ]
    input_files: dict[str, str]
    about_user_config: str
    about_model_config: str


llm = AzureChatOpenAI(
    azure_endpoint=AzureOpenAICFG.ENDPOINT,
    azure_deployment=AzureOpenAICFG.API_DEPLOYMENT,
    api_key=AzureOpenAICFG.API_KEY,
    api_version=AzureOpenAICFG.API_VERSION,
    max_retries=AzureOpenAICFG.MAX_RETRIES,
).bind_tools(tools)


async def gpt(state: State):
    message = await llm.ainvoke(state["messages"])

    logger.info(f"GPT node invoked. Response:\n{message.json()}")

    return {"messages": message}


async def preprocess(state: State):
    human_message_content = [
        TextContent(
            text=f"""[{datetime.datetime.now(CoreCFG.TIME_ZONE).strftime("%Y-%m-%d %H:%M:%S")}]"""
        )
    ]
    input_files = state["input_files"] if state["input_files"] else {}

    for content in state["content"]:
        if isinstance(content, FileContent):
            file_name = content.file_url.url.split("/")[-1]
            local_file_path = download_file(
                content.file_url.url, state["conversation_id"]
            )

            input_files[file_name] = local_file_path

            content = TextContent(text=f"[File] {file_name}")

        human_message_content.append(content)

    if not state["messages"]:
        about_user_config = state["about_user_config"]
        about_model_config = state["about_model_config"]

        if about_user_config or about_model_config:
            chatgpt_system_prompt = customize_prompt(
                about_user_config, about_model_config
            )
        else:
            chatgpt_system_prompt = CHATGPT_SYSTEM_PROMPT

        messages = [
            SystemMessage(content=chatgpt_system_prompt),
            HumanMessage(content=human_message_content),
        ]
    else:
        messages = [HumanMessage(content=human_message_content)]
        about_user_config = ""
        about_model_config = ""

    return {
        "messages": messages,
        "input_files": input_files,
        "about_user_config": about_user_config,
        "about_model_config": about_model_config,
    }


def postprocess(state: State):
    message = state["messages"][-1]
    # url_regex = r'(http?://[^\s]+)'
    # message.content = re.sub(r"\*\*.*?\*\*", "", str(message.content))
    # message.content = re.sub(r"\[.*?\]", "", str(message.content))
    # message.content = re.sub(r"\(.*?\)", "", str(message.content))

    return {"messages": message}


chat_summarizer = AzureChatOpenAI(
    azure_endpoint=AzureOpenAICFG.ENDPOINT,
    azure_deployment=AzureOpenAICFG.API_DEPLOYMENT,
    api_key=AzureOpenAICFG.API_KEY,
    api_version=AzureOpenAICFG.API_VERSION,
    max_retries=AzureOpenAICFG.MAX_RETRIES,
)


async def summarize(state: State):
    summary = ""

    if not state["messages"]:
        content = "\n".join(
            message.text if isinstance(message, TextContent) else "[File/Image]"
            for message in state["content"]
        )

        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=f"Summarize this query:\nUser: \n{content}"),
        ]

        summary = await chat_summarizer.ainvoke(messages)
        summary = summary.content

        if summary.endswith((".", ("?"), ("!"))):
            summary = summary[:-1]

    return {"thread_name": summary}
