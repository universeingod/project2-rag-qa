import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent        # project2-rag-qa/ 目录
sys.path.insert(0, str(BASE.parent))          # 把 projects/ 加进搜索路径
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding_demo import embed
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel


# 一次性初始化（服务启动时只跑一次）
load_dotenv()
client1 = chromadb.PersistentClient(path=str(BASE / "chroma_db"))
client2 = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

app = FastAPI()


# 请求体：别人发来的数据长什么样
class BuildRequest(BaseModel):
    pdf_path : str

class AskRequest(BaseModel):
    query : str
# 建库：抽文本 + 按页切块 + 向量化 + 入库，返回 chunk 总数
def build_index(pdf_path):
    reader = PdfReader(pdf_path)
    print(f"共 {len(reader.pages)} 页")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 40,
        separators = ["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = []
    chunk_pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text.strip():
            continue
        for c in splitter.split_text(page_text):
            chunks.append(c)
            chunk_pages.append(i + 1)
    print(f"共切成 {len(chunks)} 块")

    # 删旧库重建，保证每次 build 都是干净的全新库（不用判断 count()==0）
    try:
        client1.delete_collection("project2")
    except Exception:
        pass
    collection = client1.get_or_create_collection(
        name = "project2",
        metadata = {"hnsw:space" : "cosine"},
    )
    collection.add(
        ids = [f"chunk_{i}" for i in range(len(chunks))],
        documents = chunks,
        embeddings = [embed(c) for c in chunks],
        metadatas = [{"page" : chunk_pages[i], "index" : i} for i in range(len(chunks))],
    )
    return len(chunks)


# 检索：拿 top-k 片段 + 它们的元数据
def top_K(query, k = 3):
    collection = client1.get_or_create_collection(name="project2")  # 数据已在磁盘，只抓把手
    qvec = embed(query)
    res = collection.query(query_embeddings=[qvec], n_results=k)
    return res["documents"][0], res["metadatas"][0]


# 生成：检索片段拼 prompt 交给 DeepSeek，返回 (答案, 来源页码列表)
def ask(query):
    search, metas = top_K(query)
    messages = [
        {"role" : "system", "content" : "你只根据提供的资料回答，资料里没有就直说不知道，不要编。"},
    ]
    context = "\n".join(f"[来源{i + 1} | 第{meta['page']}页]\n{doc}" for i, (doc, meta) in enumerate(zip(search, metas)))
    messages.append({"role" : "user", "content" : f"资料片段: \n{context}\n\n问题:{query}\n请基于资料回答，并标注答案来自哪个来源。"})
    response = client2.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
        stream=False,
    )
    answer = response.choices[0].message.content
    sources = [{"page": meta["page"]} for meta in metas]  # 只留 page，去掉 index，匹配响应格式
    return answer, sources


# 接口 1：建库
@app.post("/build_index")
def api_build(req : BuildRequest):
    return {"status" : "ok", "chunks" : build_index(req.pdf_path)}
# 接口 2：问答
@app.post("/ask")
def api_ask(req : AskRequest):
    answer, sources = ask(req.query)
    return {"answer" : answer, "sources" : sources}

@app.get("/health")
def health_check():
    return {"status" : "ok"}