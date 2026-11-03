import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)

def chat_node(state: MessagesState):
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}

g = StateGraph(MessagesState)
g.add_node("chat", chat_node)
g.add_edge(START, "chat")
g.add_edge("chat", END)

config = {"configurable": {"thread_id": "user-001"}}

# ---------- 测试开关 ----------
FIRST_RUN = False
   # 第一次跑设 True；验证恢复时改成 False

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = g.compile(checkpointer=checkpointer)

    if FIRST_RUN:
        # 第 1 轮：自我介绍（存入记忆）
        r1 = app.invoke(
            {"messages": [("user", "我叫毕总，我最喜欢的数字是 7")]}, config
        )
        print("第1轮：", r1["messages"][-1].content, "\n")

    # 第 2 轮：考记忆
    r2 = app.invoke(
        {"messages": [("user", "我叫什么名字？喜欢的数字是多少？")]}, config
    )
    print("第2轮：", r2["messages"][-1].content)
