from typing import Annotated
from datetime import datetime

from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import InjectedState

from src.utils.logger import logger
from src.module.tool.code_interpreter import execute

from src.config.constant import QueryVideosCFG


@tool
def query_restaurant(
    prompt: str,
):
    """Query for restaurant in VectorDB

    Args:
        prompt (str): User's query for restaurant

    Returns:
        Top 5 restaurant match with user's query
    """
    start_time_timestamps = str(int(datetime_str_to_timestamp(start_time)))
    end_time_timestamps = str(int(datetime_str_to_timestamp(end_time)))
    payload = {}
    url = f"{QueryVideosCFG.ENDPOINTS}?prompt={prompt}&start_time={start_time_timestamps}&end_time={end_time_timestamps}&num_results={num_results}"
    logger.info(url)
    result = request_query_api(url=url)
    return result


tools = [query_restaurant]
