import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# 1. 定义状态：所有节点共享的数据结构
class State(TypedDict):
    messages: list

# 2. 初始化模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)

# 3. 定义节点：接收状态，返回增量更新
def agent_node(state: State):
    history = "\n".join(state["messages"])
    reply = llm.invoke(history).content
    return {"messages": state["messages"] + [reply]}

# 4. 建图：节点 + 边
g = StateGraph(State)
g.add_node("agent", agent_node)
g.add_edge(START, "agent")      # 入口
g.add_edge("agent", END)        # 出口
app = g.compile()

# 5. 运行
result = app.invoke({"messages": ["用一句话解释什么是 LangGraph"]})
print(result["messages"][-1])
