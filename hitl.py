import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)

class State(TypedDict):
    task: str
    draft: str

# 节点1：Agent 生成草稿
def write_node(state: State):
    draft = llm.invoke(f"写一条 50 字内的朋友圈，主题：{state['task']}").content
    return {"draft": draft}

# 节点2：人工审批 —— 图在这里暂停
def approval_node(state: State):
    # interrupt() 让整个图暂停，把问题抛给外部
    decision = interrupt({"draft": state["draft"], "question": "批准发布吗？(yes/no)"})
    return {"draft": state["draft"] + f" [审批结果: {decision}]"}

# 节点3：发布（模拟）
def publish_node(state: State):
    print("\n>>> 已发布：", state["draft"])

g = StateGraph(State)
g.add_node("write", write_node)
g.add_node("approval", approval_node)
g.add_node("publish", publish_node)
g.add_edge(START, "write")
g.add_edge("write", "approval")
g.add_conditional_edges(
    "approval",
    lambda s: END if "no" in s["draft"] else "publish",  # 拒绝就终止
)
g.add_edge("publish", END)

# 注意：HITL 必须配 checkpointer（暂停/恢复依赖状态存储）
app = g.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "post-001"}}

# ---------- 第一次调用：跑到 approval 处暂停 ----------
result = app.invoke({"task": "周末去爬山"}, config)
print("=== 图已暂停，等待审批 ===")
print(result["__interrupt__"][0].value["draft"])

# ---------- 模拟人工看草稿后批准 ----------
answer = input("\n批准发布吗？(yes/no): ")

# ---------- 恢复执行：把人的决定传回去 ----------
final = app.invoke(Command(resume=answer), config)
print("\n最终状态：", final["draft"])
