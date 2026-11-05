1.**概述**：一个本地文档 RAG 问答系统。把 PDF 白皮书按页切分、向量化存入 Chroma，检索 top-k 片段交给 DeepSeek 生成带页码引用的答案；用 FastAPI 暴露建库/问答两个接口，system prompt 强制「资料没有就直说」来防幻觉。

2.**功能清单**：PDF 按页切分入库 / 向量检索 top-k / 生成带引用页码的答案 / 防幻觉（资料外直说不知道）/ FastAPI 接口（build_index + ask + health）

3.**技术栈**：FastAPI · Chroma · DashScope Embedding · DeepSeek API · LangChain TextSplitter · pypdf · python-dotenv

4.**架构图**：

```mermaid
flowchart TB
    C[客户端 test_rag.py] -->|POST /build_index| B[FastAPI /build_index]
    C -->|POST /ask| A[FastAPI /ask]
    B --> P[(白皮书 PDF)]
    P --> S[切分 chunk=300/overlap=40]
    S --> E[向量化 embed]
    E --> DASH[DashScope Embedding]
    E --> DB[(Chroma 磁盘 chroma_db/)]
    A --> R[检索 top-3]
    R --> E
    R --> DB
    R --> G[拼 prompt 生成]
    G --> DS[DeepSeek 生成]
    G --> C
```

5.**截图**

![问答示例](images/查找pdf文件的一个问题.png)

6.**快速开始**：

```bash
pip install -r requirements.txt
# 复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY
uvicorn main:app --reload
```

另开一个终端建库（首次跑一次即可）：

```bash
python -c "import requests; print(requests.post('http://127.0.0.1:8000/build_index', json={'pdf_path': 'docs/企业RAG知识库建设与使用白皮书.pdf'}).json())"
```

然后问答：

```bash
python test_rag.py
```
