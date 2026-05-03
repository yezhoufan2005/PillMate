# PillMate — 家庭用药多模态 RAG 系统

基于 **CLIP 视觉编码 + Faiss 向量检索 + 大语言模型生成** 的家庭用药智能助手。上传药片图片或描述症状，系统自动识别药物并基于 FDA 官方说明书提供专业用药建议。

**技术栈**: Python · FastAPI · CLIP ViT-B/32 · Faiss · ZhipuAI GLM-4-Flash · PyTorch

---

## 项目结构

```
MRAG/
├── app.py                     # FastAPI 服务入口
├── requirements.txt           # Python 依赖
├── .env                       # 环境变量 (API Key)
├── knowledge_base/            # 【核心数据 - 不可删除】
│   ├── database.json          # 894 种药物的结构化知识库
│   ├── images/                # 5411 张药片图片 (按 NDC 分目录)
│   └── README.md
├── mrag/                      # 核心引擎
│   ├── engine.py              # 多模态编码 + Faiss 索引 + 检索 + LLM 生成
│   ├── config.py              # 路径/模型配置
│   └── data_loader.py         # 知识库加载适配器
├── mrag_output/               # 运行时输出 (自动生成)
│   ├── data_storage/          # Faiss 索引 + SQLite 数据库
│   ├── logs/                  # 运行日志
│   └── temp_uploads/          # 临时上传图片
├── static/
│   └── index.html             # 前端聊天界面
└── test/                      # 测试脚本
    ├── run_all.py                 # 一键运行入口
    ├── test_retrieval.py          # 检索精度 (26 项)
    ├── test_image_recognition.py  # 图片识别 (14 项)
    └── test_scenarios.py          # 全场景功能 (15 场景)
```

---

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
ZHIPUAI_API_KEY=你的智谱AI密钥
```

### 3. 启动服务

```bash
python app.py
```

首次启动会自动加载知识库并构建 Faiss 索引（894 种药物，约 3 分钟 CPU 编码）。之后启动直接加载已有索引，秒级就绪。

### 4. 打开前端

浏览器访问 `http://localhost:8000`

---

## 系统架构

```
用户输入 (文字/图片)
        │
        ▼
   ┌─────────────┐
   │  FastAPI     │ ← app.py: 路由 + 查询类型判断 + symptom→术语映射
   └──────┬──────┘
          │
   ┌──────▼──────────────────────────────────┐
   │  engine.py (Multimodal RAG Pipeline)     │
   │                                          │
   │  MultimodalEncoder (CLIP ViT-B/32)       │
   │       ├── 文本 → text_vector (512d)       │
   │       └── 图片 → image_vector (512d)      │
   │                                          │
   │  Indexer (Faiss IndexIDMap2 + IndexFlatIP)│
   │       ├── text_index  (894 vectors)       │
   │       ├── image_index (894 vectors)       │
   │       └── mean_index  (894 vectors)       │
   │                                          │
   │  Retriever                               │
   │       ├── 纯文本 → text_index             │
   │       ├── 纯图片 → image_index (k=8)      │
   │       └── 图文混合 → 双路检索 (图×0.85 + 文×0.15) │
   │                                          │
   │  Generator (ZhipuAI GLM-4-Flash)         │
   │       ├── 动态 System Prompt (text/image) │
   │       ├── temperature=0.3 (偏确定性)       │
   │       ├── max_tokens=3000                │
   │       └── 免责声明自动追加                 │
   └──────────────────────────────────────────┘
```

### 检索策略

| 查询类型 | 策略 |
|---------|------|
| 纯文本 | 文本向量检索 Faiss text_index → Top-K |
| 纯图片 | 图片向量检索 Faiss image_index → Top-8 |
| 图文混合 | 图片检索 (×0.85) + 文本检索 (×0.15) 加权合并 → Top-K |
| 多图上传 | 每张图独立检索，max-pooling 合并，再与文本加权混合 |

### System Prompt 动态适配

- **图片模式**: 禁止 LLM 说 "我看不到图片"，强制基于视觉匹配结果输出
- **文本模式**: 使用 "推荐药物" 格式（非 "识别结果"），支持症状→适应症匹配
- **症状查询**: 中文症状词自动映射为医学术语 (如 `头痛→analgesic;pain;migraine`)

---

## API 接口

### `GET /health`
健康检查，返回索引状态。

```json
{"status": "ready", "documents": 894}
```

### `POST /query`
提交查询。

| 参数 | 类型 | 说明 |
|------|------|------|
| `query_text` | str | 文字问题 |
| `images` | File[] | 药片图片 (支持多张) |
| `top_k` | int | 返回结果数 (默认 5) |

返回:
```json
{
  "success": true,
  "query_type": "image|text|multimodal",
  "response": "Markdown 格式的药品信息...",
  "retrieved_drugs": [
    {"ndc": "00093-0148-01", "drug_name": "Naproxen", "similarity": 0.795, ...}
  ]
}
```

---

## 知识库构建流程

> 知识库已构建完毕 (`knowledge_base/database.json`)，以下为可复现流程：

1. **NDC 提取**: 从 ePillID 数据集提取 928 个唯一 NDC 码
2. **NDC → RxNorm 映射**: 通过 NIH RxNorm API 将过期 NDC 映射为 rxcui (成功率 94.2%)
3. **RxNorm → DailyMed SPL**: 通过 rxcui 获取 FDA 官方药品说明书 XML
4. **SPL 解析**: 提取 HOW SUPPLIED (外观属性) + 9 章完整说明书内容
5. **结构化输出**: 预处理 rag_text 字段以供 CLIP 编码

详细说明见 `knowledge_base/实验报告_知识库构建.md`

---

## 测试

```bash
# 一键运行所有测试
python test/run_all.py

# 或单独运行
python test/test_retrieval.py           # 检索精度: 文本10项 + 图像6项 + 混合2项
python test/test_image_recognition.py   # 图片识别: 禁词检查 + 多图 + 用药判断
python test/test_scenarios.py           # 全场景: 症状推荐/药效判断/禁忌排查等15场景
```

### 测试覆盖

| 脚本 | 覆盖范围 | 测试项 |
|------|---------|:-----:|
| `test_retrieval.py` | Faiss 向量检索精度 | 18 |
| `test_image_recognition.py` | 图片识别禁止词 + 回答质量 | 14 |
| `test_scenarios.py` | 端到端家庭用药场景 | 15 场景 × 多项检查 |

---

## 依赖

| 库 | 用途 |
|----|------|
| `faiss-cpu` | 向量检索引擎 |
| `transformers` | CLIP 模型加载 |
| `torch` | 深度学习推理 |
| `Pillow` | 图像处理 |
| `zhipuai` | LLM API 调用 |
| `fastapi` + `uvicorn` | Web 服务框架 |
| `python-multipart` | 文件上传 |
| `python-dotenv` | 环境变量 |

---

## 许可证

本项目仅用于学术研究和课程设计目的。药品信息来源于 FDA DailyMed SPL，图片来源于 ePillID 数据集。
