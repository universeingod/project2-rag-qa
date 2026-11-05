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


#加载 + 分割
reader = PdfReader(BASE / "docs"/"企业RAG知识库建设与使用白皮书.pdf")
print(f"共 {len(reader.pages)} 页")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 120,
    chunk_overlap = 20,
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
#新建向量库+文本向量化进向量库
client1 = chromadb.PersistentClient(path=str(BASE / "chroma_db"))
collection = client1.get_or_create_collection(
    name = "project2",
    metadata = {"hnsw:space" : "cosine"},
)
if collection.count() == 0:
    collection.add(
        ids = [f"chunk_{i}" for i in range(len(chunks))],
        documents = chunks,
        embeddings = [embed(c) for c in chunks],
        metadatas = [{"page" : chunk_pages[i], "index" : i} for i in range(len(chunks))],
    )
#检索与生成
def top_K(query, k = 3):
    qvec = embed(query)
    res = collection.query(query_embeddings=[qvec], n_results=k)
    return res["documents"][0], res["metadatas"][0]

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
client2 = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


while True:
    try:
        user_ask = input("从文档查找你的问题...")
        if user_ask.strip() == "退出查找":
            break
    except (EOFError, KeyboardInterrupt):
        print("输入结束")
        break
    if not user_ask:
        continue
    search, metas = top_K(user_ask)
    try:
        messages = [
            {"role" : "system", "content" : "你只根据提供的资料回答，资料里没有就直说不知道，不要编。"},
        ]
        context = "\n".join(f"[来源{i + 1} | 第{meta['page']}页]\n{doc}" for i, (doc, meta) in enumerate(zip(search, metas)))
        messages.append({"role" : "user", "content" : f"资料片段: \n{context}\n\n问题:{user_ask}\n请基于资料回答，并标注答案来自哪个来源。"})
        response = client2.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
            stream=False,
        )
        answer = response.choices[0].message.content
        print(answer)
    except Exception as e:
        print(f"请求出错, {e}")
