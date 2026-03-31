from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.cuda.amp

# 初始化 FastAPI
app = FastAPI(title="BGE Reranker API (RTX5060Ti Optimized)", version="v1")

# ====================== 你的本地模型路径 ======================
MODEL_PATH = r"I:\AI\APP\reranker\models\BAAI\bge-reranker-v2-m3"
# ==============================================================

# ====================== RTX5060Ti 16G 优化配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024  # 最大上下文长度（官方上限）
BATCH_MAX_DOCS = 30  # 支持知识库最大30篇文档
# ====================================================================

# 加载模型 + GPU 加速 + 半精度（显存极低）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model = model.half().to(DEVICE)  # FP16 + 显卡加速
model.eval()

# ====================== 模型列表接口（Cherry Studio 必需） ======================
@app.get("/rerank/v1/models")
async def list_models():
    return {
        "data": [{
            "id": "bge-reranker-v2-m3",
            "object": "model",
            "created": 1720000000,
            "owned_by": "BAAI",
            "permission": []
        }],
        "object": "list"
    }

# ====================== 重排请求格式 ======================
class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model: str = "bge-reranker-v2-m3"
    top_n: int | None = None

# ====================== 批量优化 + GPU 高速重排接口 ======================
@app.post("/rerank/v1/rerank")
async def rerank(request: RerankRequest):
    # 限制最大30篇文档（匹配知识库）
    docs = request.documents[:BATCH_MAX_DOCS]
    sentence_pairs = [[request.query, doc] for doc in docs]

    # GPU 混合精度推理（极速 + 省显存）
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        inputs = tokenizer(
            sentence_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_LENGTH
        ).to(DEVICE)

        # 推理得分
        scores = model(**inputs).logits.squeeze(-1).float().cpu().numpy()

    # 清理显存
    del inputs
    torch.cuda.empty_cache()

    # 排序结果
    results = [
        {"index": i, "score": float(score), "text": docs[i]}
        for i, score in enumerate(scores)
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    
    if request.top_n:
        results = results[:request.top_n]

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    # 本地安全运行
    uvicorn.run(app, host="127.0.0.1", port=8000)