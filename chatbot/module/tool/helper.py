import json
from typing import Union, Any, Literal

from langchain_core.messages import ToolMessage, AnyMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

from src.module.graph.nodes import State
from src.utils.logger import logger


def handle_tool_error(state: State) -> dict:
    conversation_id = state["conversation_id"]
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    error_messages = [
        ToolMessage(
            content=f"Error: {repr(error)}\n please fix your mistakes.",
            tool_call_id=tc["id"],
        )
        for tc in tool_calls
    ]

    logger.error(
        f"Conversation ID: {conversation_id} | Error Messages:\n{json.dumps(error_messages, indent=4)}"
    )

    return {"messages": error_messages}


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any]],
) -> Literal["tools", "postprocess"]:
    """Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    Handles some errors when calling tools from LLM.

    Args:
        state (Union[list[AnyMessage], dict[str, Any]]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.


    Examples:
        Create a custom ReAct-style agent with tools.

        ```pycon
        >>> from langchain_anthropic import ChatAnthropic
        >>> from langchain_core.tools import tool
        ...
        >>> from langgraph.graph import StateGraph
        >>> from langgraph.prebuilt import ToolNode, tools_condition
        >>> from langgraph.graph.message import add_messages
        ...
        >>> from typing import TypedDict, Annotated
        ...
        >>> @tool
        >>> def divide(a: float, b: float) -> int:
        ...     \"\"\"Return a / b.\"\"\"
        ...     return a / b
        ...
        >>> llm = ChatAnthropic(model="claude-3-haiku-20240307")
        >>> tools = [divide]
        ...
        >>> class State(TypedDict):
        ...     messages: Annotated[list, add_messages]
        >>>
        >>> graph_builder = StateGraph(State)
        >>> graph_builder.add_node("tools", ToolNode(tools))
        >>> graph_builder.add_node("chatbot", lambda state: {"messages":llm.bind_tools(tools).invoke(state['messages'])})
        >>> graph_builder.add_edge("tools", "chatbot")
        >>> graph_builder.add_conditional_edges(
        ...     "chatbot", tools_condition
        ... )
        >>> graph_builder.set_entry_point("chatbot")
        >>> graph = graph_builder.compile()
        >>> graph.invoke({"messages": {"role": "user", "content": "What's 329993 divided by 13662?"}})
        ```
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if (
        hasattr(ai_message, "invalid_tool_calls")
        and len(ai_message.invalid_tool_calls) > 0
    ):
        for itc in ai_message.invalid_tool_calls:
            args = itc["args"]

            if itc["name"] == "execute_code" or itc["name"] == "python":
                if args.startswith("python"):
                    args.removeprefix("python")

                if "code_block" not in args:
                    args = {"code_block": args}

            tc = {
                "name": "execute_code",
                "args": args,
                "id": itc["id"],
                "type": "tool_call",
            }

            ai_message.tool_calls.append(tc)

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "postprocess"
