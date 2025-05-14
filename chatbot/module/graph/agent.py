from langgraph.graph import StateGraph, START, END

from src.module.graph.nodes import State, gpt, preprocess, summarize, tools, postprocess
from src.module.tool.helper import create_tool_node_with_fallback, tools_condition

graph_bulder = StateGraph(State)

graph_bulder.add_node("gpt", gpt)
graph_bulder.add_node("preprocess", preprocess)
graph_bulder.add_node("postprocess", postprocess)
graph_bulder.add_node("summarize", summarize)
graph_bulder.add_node("tools", create_tool_node_with_fallback(tools))

graph_bulder.add_edge(START, "preprocess")
graph_bulder.add_edge(START, "summarize")
graph_bulder.add_edge("preprocess", "gpt")
graph_bulder.add_conditional_edges("gpt", tools_condition)
graph_bulder.add_edge("tools", "gpt")
graph_bulder.add_edge("postprocess", END)
graph_bulder.add_edge("summarize", END)
