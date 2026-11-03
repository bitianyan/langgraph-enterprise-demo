import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ---------- 1. 知识库（模拟企业文档） ----------
DOCS = [
    "公司年假制度：入职满 1 年 5 天，满 3 年 10 天，满 5 年 15 天。",
    "报销流程：发票需在费用发生后 30 天内提交，超过 5000 元需部门总监审批。",
    "考勤规定：工作日 9:00-18:00，迟到 3 次以内不扣款，超过按 50 元/次。",
    "远程办公：每周可申请 2 天远程，需提前 1 天在 OA 系统报备。",
    "社保公积金：入职当月开始缴纳，公积金比例为工资的 12%。",
    "加班制度：加班需审批，工作日加班按 1.5 倍调休，节假日 3 倍。",
    "试用期：合同 3 年制员工试用期 6 个月，表现优秀可提前转正。",
    "培训基金：每年每人 2000 元学习预算，可用于买书、报课程。",
    "差旅标准：一线城市酒店上限 600 元/晚，二线城市 400 元/晚。",
    "晋升周期：每年 4 月和 10 月两次晋升评审窗口。",
]

# ---------- 2. 建向量库（本地 embedding，内存模式） ----------
client = QdrantClient(":memory:")
client.create_collection("kb", vectors_config=VectorParams(size=512, distance=Distance.COSINE))

model = TextEmbedding("BAAI/bge-small-zh-v1.5")   # 首次运行会下载约 100MB
vecs = list(model.embed(DOCS))
client.upsert("kb", points=[
    PointStruct(id=i, vector=v.tolist(), payload={"text": DOCS[i]})
    for i, v in enumerate(vecs)
])

def retrieve(query: str, top_k: int = 2) -> list:
    qvec = list(model.embed([query]))[0].tolist()
    hits = client.query_points("kb", query=qvec, limit=top_k).points
    return [h.payload["text"] for h in hits]

# ---------- 3. 图结构：检索 → 质检 → 生成 ----------
class State(TypedDict):
    question: str
    docs: list
    answer: str

def retrieve_node(state):
    return {"docs": retrieve(state["question"])}

def grade_node(state):
    """质检检索结果：不相关则改写查询重试一次（防幻觉核心）"""
    docs = "\n".join(state["docs"])
    verdict = llm.invoke(
        f"判断以下资料能否回答问题，只答 能 或 不能。\n资料：{docs}\n问题：{state['question']}"
    ).content
    if "不能" in verdict:
        rewritten = llm.invoke(
            f"改写这个问题使它更适合公司制度检索，只输出改写后的问题：{state['question']}"
        ).content
        return {"docs": retrieve(rewritten), "question": rewritten}
    return {}

def generate_node(state):
    docs = "\n".join(state["docs"])
    answer = llm.invoke(
        f"仅根据以下资料回答，资料中没有的信息就回答'资料中未提及'，不要编造。\n资料：\n{docs}\n\n问题：{state['question']}"
    ).content
    return {"answer": answer}

g = StateGraph(State)
g.add_node("retrieve", retrieve_node)
g.add_node("grade", grade_node)
g.add_node("generate", generate_node)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", lambda s: END if not s.get("docs") else "generate")
g.add_edge("generate", END)
app = g.compile()

# ---------- 4. 测试 ----------
for q in ["年假有几天？", "出差住酒店能报多少？", "公司篮球队什么时候训练？"]:
    print("=" * 40, "\n问题：", q)
    r = app.invoke({"question": q})
    print("回答：", r["answer"][:150], "\n")
