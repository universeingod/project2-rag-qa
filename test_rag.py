import requests
"""
r = requests.post(
    "http://127.0.0.1:8000/build_index",
    json={"pdf_path": "docs/企业RAG知识库建设与使用白皮书.pdf"},
)
print(r.json())   # 应该看到 {"status": "ok", "chunks": 101}

r = requests.get("http://127.0.0.1:8000/health")
print(r.json())
"""
questions = ["上下文增强向量结合 BM25 能把检索失败率降低多少？", "BEIR 数据集主要用来做什么？", "企业 RAG 检索验证要注意哪些坑？",
             "企业 RAG 知识库文档分块时，白皮书推荐优先考虑哪些因素？", "RAG 系统里重排模型一般放在哪个环节，它解决什么问题？",
             "企业搭建 RAG 知识库时，文档预处理包含哪些步骤？", "怎么判断 RAG 输出结果是否存在幻觉，白皮书给出了哪些核验思路？",
             "企业 RAG 知识库权限隔离一般有哪几种实现方案？", "基于向量检索的 RAG，冷启动阶段容易遇到什么问题？",
             "LLM是啥？"
             ]
for q in questions:
    r = requests.post("http://127.0.0.1:8000/ask", json={"query" : q})
    data = r.json()
    print("问", q)
    print("答", data["answer"])
    print("来源", data["sources"])
    print("----------")
