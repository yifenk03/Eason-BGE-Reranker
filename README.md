---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: "00000000000000000000000000000000"
    PropagateID: "00000000000000000000000000000000"
    ReservedCode1: 304402203a116725bfa3945398d81e09a70fd3d0a33bc5ed5f2aeb0fbbca21d0f775b47c02200cfcdd15feded148e0ad067371ebb16f3b6911e0ba7de889060babc845571e32
    ReservedCode2: 3046022100bc16479e10fae444c37e1cbbc444b56f3a6e746cd29d218176cfa39a0f6f04d1022100989a2a37b704f3cee9bdb775644e75f781458ce721715be974e6602a7d8f7279
---

# BGE Reranker API

基于 FastAPI 的本地 BGE 重排序模型推理服务，针对 RTX 5060 Ti 16G 显存进行了优化。

## 功能特性

- **本地部署**：无需云端服务，保护数据隐私
- **GPU 加速**：支持 CUDA 加速推理
- **混合精度**：采用 FP16 半精度，大幅降低显存占用
- **批量处理**：支持批量文档重排序，最多 30 篇文档
- **兼容性强**：符合 OpenAI-style API 格式，便于集成

## 环境要求

| 项目 | 最低要求 |
|------|----------|
| Python | >= 3.8 |
| 显存 | 16GB (RTX 5060 Ti 或同等规格) |
| 操作系统 | Windows / Linux / macOS |
| CUDA | 11.8+ (用于 GPU 加速) |

## 安装步骤

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd bge_rerank_api
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载模型

将 BGE Reranker 模型下载到本地，修改 `bge_rerank_api.py` 中的 `MODEL_PATH`：

```python
MODEL_PATH = r"I:\AI\APP\reranker\models\BAAI\bge-reranker-v2-m3"
```

模型下载地址：
- Hugging Face: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Modelscope: https://www.modelscope.cn/models

## 使用方法

### 启动服务

```bash
python bge_rerank_api.py
```

服务将在 `http://127.0.0.1:8000` 启动。

### API 接口

#### 1. 获取模型列表

```
GET /rerank/v1/models
```

响应示例：
```json
{
  "data": [{
    "id": "bge-reranker-v2-m3",
    "object": "model",
    "created": 1720000000,
    "owned_by": "BAAI",
    "permission": []
  }],
  "object": "list"
}
```

#### 2. 文本重排序

```
POST /rerank/v1/rerank
```

请求体：
```json
{
  "query": "什么是人工智能？",
  "documents": [
    "人工智能是计算机科学的一个分支。",
    "机器学习是人工智能的子领域。",
    "今天天气不错。"
  ],
  "model": "bge-reranker-v2-m3",
  "top_n": 2
}
```

响应示例：
```json
{
  "results": [
    {"index": 0, "score": 0.95, "text": "人工智能是计算机科学的一个分支。"},
    {"index": 1, "score": 0.87, "text": "机器学习是人工智能的子领域。"}
  ]
}
```

### 客户端调用示例

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/rerank/v1/rerank",
    json={
        "query": "如何学习Python？",
        "documents": [
            "Python是一种高级编程语言。",
            "学习Python需要多实践。",
            "JavaScript主要用于前端开发。"
        ],
        "top_n": 2
    }
)
print(response.json())
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | - | 模型本地路径（必须配置） |
| `MAX_LENGTH` | 1024 | 最大上下文长度 |
| `BATCH_MAX_DOCS` | 30 | 单次请求最大文档数 |
| `DEVICE` | auto | 设备选择（cuda/cpu） |

## 性能优化

本项目针对 RTX 5060 Ti 16G 进行了以下优化：

1. **FP16 混合精度**：将模型权重转换为半精度，大幅降低显存占用
2. **批量处理**：支持批量文档处理，提高吞吐量
3. **显存清理**：推理后自动清理显存，防止内存泄漏
4. **CUDA 加速**：启用 GPU 加速，提升推理速度

## 与 Cherry Studio 集成

本 API 兼容 Cherry Studio 的重排序功能配置：

1. 打开 Cherry Studio 设置
2. 进入「模型服务」→「重排模型」
3. 配置 API 地址：`http://127.0.0.1:8000`
4. 选择模型：`bge-reranker-v2-m3`

## 常见问题

### Q: 显存不足怎么办？

A: 尝试减小 `BATCH_MAX_DOCS` 或降低 `MAX_LENGTH`。

### Q: 如何指定其他模型？

A: 下载其他 BAAI 重排序模型，修改 `MODEL_PATH` 路径和 `list_models` 中的模型 ID。

### Q: 支持 CPU 运行吗？

A: 支持，将自动回退到 CPU 模式，但推理速度会显著降低。

## 许可证

MIT License

## 参考链接

- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Transformers 文档](https://huggingface.co/docs/transformers/)
