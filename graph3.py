import os
import json
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

class State(TypedDict):
    messages: Annotated[list, operator.add]  # add：各节点返回值自动拼接

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ---------- 工具定义 ----------
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

def get_weather(city: str) -> str:
    """查询天气（模拟数据）"""
    return json.dumps({"city": city, "weather": "晴", "temp": "22°C"}, ensure_ascii=False)

TOOLS = {"calculator": calculator, "get_weather": get_weather}

# ---------- 节点1：Agent 决策 ----------
def agent_node(state: State):
    tool_desc = "\n".join(
        f"- {name}: {fn.__doc__}" for name, fn in TOOLS.items()
    )
    prompt = f"""你是助手，可以使用以下工具：
{tool_desc}

规则：需要用工具时，只输出一行 JSON：{{"tool": "工具名", "args": {{"参数名": "值"}}}}
不需要工具时，直接输出最终答案。

对话历史：
{chr(10).join(state['messages'])}"""
    reply = llm.invoke(prompt).content.strip()
    return {"messages": ["agent: " + reply]}

# ---------- 节点2：执行工具 ----------
def tool_node(state: State):
    last = state["messages"][-1]
    body = last.split("agent: ", 1)[1]
    try:
        call = json.loads(body)
        result = TOOLS[call["tool"]](**call["args"])
        return {"messages": [f"tool result: {result}"]}
    except Exception:
        return {"messages": []}   # 不是工具调用，交给路由判断

# ---------- 路由：还有工具调用吗 ----------
def router(state: State):
    last = state["messages"][-1]
    if last.startswith("agent: ") and last[7:].lstrip().startswith("{"):
        return "tool"
    return END

g = StateGraph(State)
g.add_node("agent", agent_node)
g.add_node("tool", tool_node)
g.add_edge(START, "agent")
g.add_conditional_edges("agent", router)   # 有工具调用 → tool 节点
g.add_edge("tool", "agent")                # 关键：工具结果 → 回到 agent，形成循环
app = g.compile()

# ---------- 测试 ----------
question = "北京今天天气怎么样？气温乘以 3 是多少？"
print(f"问题：{question}\n")
result = app.invoke({"messages": [f"user: {question}"]})
for m in result["messages"]:
    print(m)
