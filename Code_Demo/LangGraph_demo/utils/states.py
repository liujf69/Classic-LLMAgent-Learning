
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# 定义状态结构
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str        # 用户查询
    search_query: str      # 优化后的搜索查询
    search_results: str    # Tavily搜索结果
    final_answer: str      # 最终答案
    step: str             # 当前步骤

