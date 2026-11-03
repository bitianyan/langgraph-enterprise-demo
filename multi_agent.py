import os
import operator
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)

class State(TypedDict):
    task: str
    history: Annotated[list, operator.add]   # 记录每个 Agent 干了什么

def log(state, who, output):
    return {"history": [f"[{who}] {output}"]}

# ---------- 三个子 Agent ----------
def researcher(state: State):
    out = llm.invoke(f"你是调研员。搜集关于「{state['task']}」的 3 个关键事实，每个一行。").content
    return log(state, "researcher", out)

def analyst(state: State):
    context = "\n".join(state["history"])
    out = llm.invoke(f"你是分析师。基于以下调研结论，提炼 2 个洞察：\n{context}").content
    return log(state, "analyst", out)

def writer(state: State):
    context = "\n".join(state["history"])
    out = llm.invoke(f"你是作者。基于以下材料写一段 100 字总结：\n{context}").content
    return log(state, "writer", out)

# ---------- Supervisor：决定下一个谁上场 ----------
def supervisor(state: State) -> Literal["researcher", "analyst", "writer", "FINISH"]:
    done = "\n".join(state["history"])
    resp = llm.invoke(f"""任务：{state['task']}
已完成的工作：
{done}

流程要求：先 researcher 调研 → 再 analyst 分析 → 最后 writer 成文。
判断下一步该谁执行，只回答一个词：researcher / analyst / writer / FINISH""").content.strip()
    return resp if resp in ("researcher", "analyst", "writer", "FINISH") else "FINISH"

g = StateGraph(State)
g.add_node("researcher", researcher)
g.add_node("analyst", analyst)
g.add_node("writer", writer)

g.add_edge(START, "researcher")                      # 先调研
route_map = {"researcher": "researcher", "analyst": "analyst", "writer": "writer", "FINISH": END}
g.add_conditional_edges("researcher", supervisor, route_map)
g.add_conditional_edges("analyst", supervisor, route_map)
g.add_conditional_edges("writer", supervisor, route_map)
g.add_edge("researcher", "analyst")                  # 固定顺序兜底
g.add_edge("analyst", "writer")

app = g.compile()

result = app.invoke({"task": "人工智能对就业市场的影响"})
for h in result["history"]:
    print(h[:150], "\n")
