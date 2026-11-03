import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

class State(TypedDict):
    question: str
    answer: str
    route: str

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)

# 节点1：分类器 —— 判断问题类型
def classifier(state: State):
    resp = llm.invoke(
        f"判断下面问题的类型，只回答一个词：数学 或 常识\n问题：{state['question']}"
    ).content.strip()
    route = "数学" if "数学" in resp else "常识"
    return {"route": route}

# 节点2：数学专家
def math_expert(state: State):
    reply = llm.invoke(f"你是数学老师，一步步解题：{state['question']}").content
    return {"answer": reply}

# 节点3：常识专家
def general_expert(state: State):
    reply = llm.invoke(f"简洁回答：{state['question']}").content
    return {"answer": reply}

# 路由函数：读 state 决定下一跳
def router(state: State) -> Literal["math", "general"]:
    return "math" if state["route"] == "数学" else "general"

g = StateGraph(State)
g.add_node("classifier", classifier)
g.add_node("math", math_expert)
g.add_node("general", general_expert)

g.add_edge(START, "classifier")
g.add_conditional_edges("classifier", router)   # 关键：条件边
g.add_edge("math", END)
g.add_edge("general", END)
app = g.compile()

# 测试两类问题
for q in ["计算 17 * 23 等于多少？", "为什么天空是蓝色的？"]:
    print("=" * 40)
    print("问题：", q)
    result = app.invoke({"question": q})
    print("路由到：", result["route"])
    print("回答：", result["answer"][:100])
