import asyncio
import json
import os

import aio_pika
import aio_pika.abc
from aio_pika import ExchangeType
from aio_pika.abc import AbstractIncomingMessage

from src.utils.logger import logger
from src.module.rabbitmq.message_handler import chat_completion
from src.config.model import RequestMessage, ResponseMessage


global_exchange_subscribe = None
global_exchange_publish = None


async def on_message(message: AbstractIncomingMessage) -> None:
    global global_exchange_publish
    async with message.process():
        data = json.loads(message.body.decode(encoding="utf8"))

        data = await chat_completion(RequestMessage(**data))

        # Publish message
        routing_key = os.environ["ROUTING_KEY_PUBLISH"]
        await global_exchange_publish.publish(
            aio_pika.Message(body=json.dumps(data.dict()).encode(encoding="utf8")),
            routing_key=routing_key,
        )
        logger.info(f"[X] ==> Publish data to {routing_key}: {data}")

    return None


async def rabbit_process():
    connection = await aio_pika.connect(os.environ["RABBITMQ_CONNECTION_STR"])
    async with connection:
        queue_name = os.environ["QUEUE_SUBSCRIBE"]

        # Creating channel
        channel: aio_pika.abc.AbstractChannel = await connection.channel()

        global global_exchange_subscribe
        global_exchange_subscribe = await channel.declare_exchange(
            os.environ["EXCHANGE_JOB_SUBSCRIBE"], ExchangeType.TOPIC, durable=True
        )

        global global_exchange_publish
        global_exchange_publish = await channel.declare_exchange(
            os.environ["EXCHANGE_PUBLISH"], ExchangeType.TOPIC, durable=True
        )

        # Declaring queue
        queue: aio_pika.abc.AbstractQueue = await channel.declare_queue(
            queue_name, durable=True, auto_delete=False
        )
        # Binding the queue to the exchange
        await queue.bind(
            global_exchange_subscribe, routing_key=os.environ["ROUTING_KEY_SUBSCRIBE"]
        )

        await queue.consume(on_message)

        logger.info("Waiting for logs. To exit press CTRL+C")
        await asyncio.Future()
