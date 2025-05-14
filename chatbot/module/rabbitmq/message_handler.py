import os
import re
import concurrent.futures
from typing import Union

import asyncio

from src.config.constant import MongoDBCFG
from src.utils.logger import logger
from src.config.model import RequestMessage, ResponseMessage, FirstResponseMessage
from src.module.graph.agent import graph_bulder
from src.module.storage.checkpointer import AsyncMongoDBSaver


async def chat_completion(
    message: RequestMessage,
) -> Union[ResponseMessage, FirstResponseMessage]:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(os.environ["CONCURRENCY_MAX_WORKERS"])
    ) as executor:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor, lambda: asyncio.run(chat_completion_event_loop(message))
        )
        return result


async def chat_completion_event_loop(
    message: RequestMessage,
) -> Union[ResponseMessage, FirstResponseMessage]:
    try:
        logger.info(f"Request received: {message.model_dump_json(indent=4)}")

        config = {"configurable": {"thread_id": message.conversation_id}}

        async with AsyncMongoDBSaver.from_conn_info(
            host=MongoDBCFG.URI, db_name=MongoDBCFG.DB_NAME
        ) as checkpointer:
            graph = graph_bulder.compile(checkpointer=checkpointer)

            output_message = await graph.ainvoke(
                input={
                    "conversation_id": message.conversation_id,
                    "model": message.model,
                    "content": message.content,
                    "about_user_config": message.about_user_message,
                    "about_model_config": message.about_model_message,
                },
                config=config,
            )

        last_message_content = output_message["messages"][-1].content

        url_regex = r"(http?://[^\s]+\.mp4)"
        video_urls = re.findall(url_regex, last_message_content)
        video_urls = [url.rstrip(")") for url in video_urls]
        logger.warning(f"Video URLS: {video_urls}")

        last_message_content = re.sub(
            r"\*\*.*?\*\*", "", str(output_message["messages"][-1].content)
        )
        last_message_content = re.sub(r"\n\d+\.\s", "", str(last_message_content))
        last_message_content = re.sub(r"\n\n", "", str(last_message_content))
        last_message_content = re.sub(r"\n\s\n", "", str(last_message_content))
        last_message_content = re.sub(r"\[.*?\]", "", str(last_message_content))
        last_message_content = re.sub(r"\(.*?\)", "", str(last_message_content))

        content = [{"type": "text", "text": last_message_content}]

        if output_message["thread_name"]:
            response = FirstResponseMessage(
                conversation_id=message.conversation_id,
                thread_id=message.conversation_id,
                thread_name=output_message["thread_name"],
                new_thread=True,
                content=content,
                video_urls=video_urls,
            )
        else:
            response = ResponseMessage(
                conversation_id=message.conversation_id,
                thread_id=message.conversation_id,
                content=content,
                video_urls=video_urls,
            )

        logger.info(f"Response created: {response.model_dump_json(indent=4)}")

        return response

    except Exception as e:
        logger.error(f"Error exception: {e}")
        response = ResponseMessage(
            conversation_id=message.conversation_id,
            thread_id=message.conversation_id,
            content=[
                {
                    "type": "text",
                    "text": "**Error:** Something went wrong. Please try again.",
                }
            ],
        )
    return response
