import os,math
import dashscope
from dotenv import load_dotenv
from http import HTTPStatus

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def embed(text: str) -> list[float]:
    #把一句话转成向量（一串数字）
    resp = dashscope.TextEmbedding.call(model="text-embedding-v4", input=text)
    if resp.status_code == HTTPStatus.OK:
        return resp.output["embeddings"][0]["embedding"]
    else:
        raise RuntimeError(f"embedding 失败, {resp}")

def cosine(a, b):
    """手写余弦相似度 = 点积 / (模长之积)，结果越接近 1 越相似"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

if __name__ == "__main__":
    texts = ["今天的太阳晒起来很舒服", "外面阳光明媚", "我喜欢吃苹果", "我喜欢用苹果手机"]
    vecs = {t : embed(t) for t in texts}
    print(f"向量维度：{len(vecs[texts[0]])}")
    print(f"取前5个数字: {vecs[texts[0]][:5]}")
    print()
    print("「今天的太阳晒起来很舒服」vs「外面阳光明媚」:", round(cosine(vecs[texts[0]], vecs[texts[1]]), 4)) #0.672
    print("「今天的太阳晒起来很舒服」vs「我喜欢吃苹果」:", round(cosine(vecs[texts[0]], vecs[texts[2]]), 4)) #0.3932
    print("「我喜欢吃苹果」vs「我喜欢用苹果手机」:", round(cosine(vecs[texts[2]], vecs[texts[3]]), 4)) #0.7158
