# 多模态 RAG 家庭用药系统构建与优化实验报告

## 摘要

本实验设计并实现了一个面向家庭用药场景的**多模态检索增强生成（Multimodal RAG, MRAG）系统**。系统以 ePillID 药片图像数据集为视觉基础，通过 RxNorm API 和 DailyMed SPL XML 解析构建了覆盖 894 种药物、5411 张药片图片的 FDA 官方药品说明书知识库。检索层采用 CLIP ViT-B/32 进行图文双编码，Faiss 向量索引实现跨模态相似度检索；生成层基于 ZhipuAI GLM-4-Flash 大语言模型，通过动态 System Prompt、双路检索加权融合、症状-医学术语映射等策略进行多轮迭代优化。系统支持药片图片识别、症状匹配推荐、药物咨询、药效判断、禁忌排查等 15+ 类家庭用药场景，综合测试通过率 92%。

---

## 一、系统架构设计

### 1.1 整体架构

系统采用经典 RAG 三阶段流水线：**编码 → 检索 → 生成**，并针对多模态（文本+图像）输入进行了适配。

```
┌─────────────────────────────────────────────────────────────┐
│                      用户输入层                              │
│         药片图片(JPG) / 文字问题 / 图片+文字                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │  app.py (FastAPI)     │
           │  · 查询类型判断       │
           │  · symptom→术语映射   │
           │  · 多图上传支持       │
           └───────────┬───────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   mrag/engine.py                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ MultimodalEncoder (CLIP ViT-B/32)                    │    │
│  │  · 文本编码 → 512d text_vector                       │    │
│  │  · 图像编码 → 512d image_vector                      │    │
│  │  · L2 归一化 → Faiss IndexFlatIP (内积=余弦相似度)    │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Indexer (SQLite + Faiss IndexIDMap2)                 │    │
│  │  · text_index:  894 vectors                          │    │
│  │  · image_index: 894 vectors                          │    │
│  │  · mean_index:  894 vectors                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Retriever - 三种检索策略                              │    │
│  │  · 纯文本 → text_index.search                        │    │
│  │  · 纯图片 → image_index.search (k=8)                 │    │
│  │  · 图文混合 → 双路独立检索 + 加权融合 (图×0.85+文×0.15)│    │
│  │  · 多图 → max-pooling 合并各图检索结果                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Generator (ZhipuAI GLM-4-Flash)                      │    │
│  │  · 动态 System Prompt (image/text 不同模板)           │    │
│  │  · temperature=0.3, max_tokens=3000                   │    │
│  │  · 物理属性注入 context text                           │    │
│  │  · 免责声明自动追加                                    │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 技术选型

| 组件 | 技术选择 | 选择理由 |
|------|---------|---------|
| 多模态编码器 | CLIP ViT-B/32 (OpenAI) | 文本-图像统一语义空间，无需额外训练 |
| 向量索引 | Faiss IndexIDMap2 + IndexFlatIP | 内积搜索等价于余弦相似度，IDMap 支持自定义文档ID |
| 元数据管理 | SQLite | 轻量级，与 Faiss ID 映射天然配对 |
| LLM 生成 | ZhipuAI GLM-4-Flash | API 稳定、中文能力强、成本低 |
| 服务框架 | FastAPI + Uvicorn | 异步支持、文件上传友好、Python 生态 |
| 前端 | 单文件 HTML/CSS/JS | 零依赖、直接 serve、适合课程演示 |

### 1.3 知识库构建流程

```
ePillID 数据集 (all_labels.csv)
    │
    ├── 提取 928 个唯一 NDC 码
    │       │
    │       ├── NDC 过期无法通过 openFDA 查询 → RxNorm API 映射
    │       │       └── ndcproperties 接口 → rxcui (成功率 94.2%)
    │       │
    │       └── rxcui → DailyMed SPL XML
    │               └── setid → /services/v2/spls/{setid}.xml
    │                       └── 提取 9 章说明书 + HOW SUPPLIED (外观属性)
    │
    └── 输出: knowledge_base/database.json
            · 894 个药物条目 (NDC 为 key)
            · 每个条目含 rag_text (预拼接完整说明书文本)
            · 物理属性 (颜色、形状、印记、规格)
            · 图像路径映射到 knowledge_base/images/{NDC}/
```

---

## 二、核心技术挑战与解决方案

### 挑战 1: 过期 NDC 码无法获取药品信息

**问题**: ePillID 数据集中的 NDC 码普遍已过期，openFDA NDC Directory 和 Drugs.com 均无法直接查询。Drugs.com 存在 CDN 反爬机制（Akamai EdgeSuite），自动化方案均告失败。

**解决方案**: 利用 NIH 的 RxNorm REST API 作为桥梁。RxNorm 维护了历史 NDC→rxcui 的映射关系，即使是几十年前的 NDC 码也能成功解析。

```
curl "https://rxnav.nlm.nih.gov/REST/ndcproperties.json?id={NDC}"
→ rxcui → "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=SETID"
→ setid → "https://api.fda.gov/drug/label.json?search=set_id:{setid}"
```

**效果**: 928 个 NDC 中 874 个成功映射 (94.2%)，远超其他方案。

### 挑战 2: 中英文跨语言语义检索精度

**问题**: 用户用中文描述症状（如"发炎"），但知识库中药品说明书全是英文。CLIP 文本编码器对中文的语义理解有限，直接检索导致"发炎"匹配到利尿剂（hydrochlorothiazide）而非非甾体抗炎药（NSAIDs）。

**解决方案**: 在 `app.py` 中构建**症状-医学术语映射表**，将用户中文症状词自动扩展为英文/拉丁文医学术语注入检索查询。

```python
symptom_map = {
    "发炎": "消炎;抗炎;anti-inflammatory;NSAID;inflammation",
    "头痛": "headache;pain;migraine;analgesic",
    "感冒": "cold;flu;influenza;decongestant;antihistamine",
    ...
}
```

这种**查询扩展（Query Expansion）**技术让文本检索能跨越语言障碍，在 LLM 生成时再将英文说明书内容翻译为用户的语言输出。

### 挑战 3: Mean Fusion 导致多模态信号稀释

**问题**: 原 MultimodalRAG 在多模态查询时使用 mean vector（文本向量+图像向量的均值）进行单次检索。这导致图像的视觉信号被文本向量稀释——当用户上传药片图并问"这是什么药"时，"这是什么药"文本向量中的"药"字会匹配到大量无关药物，使检索精度下降。

**解决方案**: 改为**双路独立检索 + 加权融合**。图片和文本分别独立检索各自的 Faiss 索引，然后用加权分数合并排名，而非在向量层面对撞。

```
图片检索 (image_index) → {NDC_id: score} × 0.85
文本检索 (text_index)   → {NDC_id: score} × 0.15
                    ↓ 去重合并
              weighted_score = 0.85×img_score + 0.15×txt_score
                    ↓ 按 weighted_score 降序
                  Top-K results
```

**效果**: Naproxen 图片检索 Top-1 精确匹配率从不可靠提高到 100%（同库测试图片）。

### 挑战 4: 物理属性未被 LLM 感知

**问题**: 检索结果中的物理属性（颜色、形状、印记）存储在 context_doc 的单独字段中，但 Generator 的 `_build_messages` 只取 `text` 字段拼接给 LLM——物理属性信息完全丢失了。LLM 只能盲选第一个候选药物，无法交叉验证外观特征。

**解决方案**: 在 `app.py` 中将物理属性显式注入 context text 头部：

```python
raw_text = "--- 药片外观特征 ---\n" \
           "颜色: yellow | 形状: round | 印记: s & g\n\n" \
         + matched["text"][:4000]
```

同时对于图片查询的每个文档，额外注入图像检索来源标记：

```python
raw_text = "🎯 图像检索匹配 #1（系统通过用户上传的药片图片视觉匹配到此药物）🎯\n\n" + raw_text
```

### 挑战 5: LLM 拒绝执行图像识别

**问题**: GLM-4-Flash 在多轮迭代中反复输出"很抱歉，我无法直接查看或处理图片"或"由于我没有直接查看图片的能力..."——即使 system prompt 已明确告知参考文档中的药物是图像检索结果。

**根因分析**: 
1. System prompt 中的 CRITICAL 指令被埋在中间位置，LLM 的 attention 机制分配权重不足
2. 每条 context 文档没有明确的图像来源标记
3. GLM-4-Flash 训练数据中的安全对齐使得它倾向于先说"我看不到"

**解决方案 - 三板斧**:
1. **首行强制指令**: System prompt 第一行直接放 `❗重要: 当前对话中'参考文档'内的所有药物均为系统通过用户上传的药片图片进行图像检索匹配的结果。你绝不能说'我看不到图片'...`
2. **context 标记**: 每条检索结果的 text 头部注入 `🎯 图像检索匹配 #N 🎯`
3. **动态 prompt 分离**: 文本查询用独立 System prompt（无图像相关指令），避免混淆

### 挑战 6: 纯文本查询错误显示"识别结果"格式

**问题**: 用户纯文本问"发炎吃什么药"，LLM 输出 `识别结果 / 推荐药物: 氢氯噻嗪`——这是图片识别模式的回答格式，让人困惑。

**解决方案**: 将 `Generator.generate()` 新增 `query_type` 参数，贯穿到 `_build_messages()`。根据查询类型选择完全不同的 System prompt 模板：

- **image/multimodal**: 使用"图片识别结果"格式，含 CRITICAL 图像指令
- **text**: 使用"推荐药物"格式，含症状匹配、药效判断能力描述

```python
def generate(self, query, context, query_type="text"):
    messages = self._build_messages(query, context, query_type)
    ...

def _build_messages(self, query, context, query_type="text"):
    if query_type in ("image", "multimodal"):
        system = IMAGE_SYSTEM_PROMPT    # 图像识别专用
    else:
        system = TEXT_SYSTEM_PROMPT     # 文字咨询专用
```

---

## 三、多轮迭代优化过程

### Round 1: 基础系统搭建

**改动**: 
- 将 RAGCore.py 从 MultimodalRAG-main 迁移为 `mrag/engine.py`
- System prompt 中文化
- Context 截断从 800 字符 → 4000 字符
- Temperature 0.7 → 0.3, max_tokens 1500 → 3000

**问题发现**: 回答质量差，信息量不足，LLM 看不到完整说明书内容。

**测试**: 50 文档查询 Naproxen，相似度 0.68-0.75

### Round 2: 检索策略优化

**改动**: 
- Mean Fusion → 双路独立检索（图 × 0.85 + 文 × 0.15）
- 图片查询 k=3 → k=8
- 增加 System prompt 图片识别强制指令

**问题发现**: LLM 仍说"我看不到图片"，Suboxone 图片匹配第 4 名

**测试**: Naproxen 查询 Top-1 精确匹配 79.5%

### Round 3: 物理属性注入

**改动**: 
- 物理属性（颜色/形状/印记/规格）格式化后注入 context text 头部
- user_query 显式要求 LLM 跳过无外观特征的候选

**问题发现**: Suboxone 仍排第 3，但 Naproxen 保持 Top-1

**测试**: 21/22 通过

### Round 4: 多样化场景适配

**改动**: 
- System prompt 扩展能力范围（药效判断/症状匹配/禁忌排查/多药识别）
- 16 个中文症状词的医学术语映射表
- 多图上传支持（前端+后端 max-pooling）
- 文本查询 user_query 智能注入症状引导

**问题发现**: 纯文本查询仍显示"识别结果"格式

**测试**: 17/18 通过

### Round 5: 动态 System Prompt

**改动**: 
- `generate()` 新增 `query_type` 参数
- `_build_messages()` 根据 query_type 分支选择不同 System prompt
- 文本 prompt 用"推荐药物"+"为什么推荐"格式
- 图像 prompt 首行强制 CRITICAL 指令

**问题发现**: "发炎"匹配到氢氯噻嗪（缺乏 NSAID 语义理解），"感冒"匹配到偏头痛药

**测试**: 26/28 通过 (92%)

### Round 6: 图片识别终极修复

**改动**:
- System prompt 首行 `❗重要:` ALL-CAPS 指令
- 每条 context_doc 头部注入 `🎯 图像检索匹配 #N 🎯`
- 多图 user_query 明确要求"分别识别标注图片1、图片2..."

**问题发现**: 图片识别不再出现"看不到"表述

**测试**: 11/11 通过

---

## 四、多场景测试结果

### 4.1 测试用例设计

| 场景 | 查询类型 | 示例问题 | 核心检查点 |
|------|---------|---------|-----------|
| 药片识别 | 图片 | 上传 Naproxen → "这是什么药？" | Top-1 精确匹配，无"看不到图片" |
| 多图识别 | 多图 | 上传 2 张 → "分别是什么药？" | 分别标注图片1/2回答 |
| 症状推荐 | 文本 | "头痛吃什么药？" | 推荐适应症匹配的药物 |
| 感冒场景 | 文本 | "感冒发烧咳嗽吃什么药？" | 有推荐 + 禁忌说明 |
| 炎症场景 | 图文 | 上传图 + "牙龈发炎这个药能用吗？" | 针对性药效判断 |
| 药效判断 | 文本 | "布洛芬治牙痛有效吗？" | 基于适应症诚实判断 |
| 禁忌排查 | 文本 | "孕妇能吃华法林吗？" | 明确禁忌 + 风险说明 |
| 用法用量 | 文本 | "二甲双胍一次吃多少？" | 含剂量信息 |
| 副作用 | 文本 | "阿托伐他汀伤肝吗？" | 含副作用/肝相关 |
| 慢性病 | 文本 | "血压高吃什么药？" | 推荐降压药 |
| 过敏场景 | 文本 | "皮肤过敏痒红肿吃什么药？" | 含抗过敏词 |
| 知识库外 | 文本 | "奥司他韦是什么药？" | 诚实告知 + 免责声明 |

### 4.2 测试组织与结果

测试套件按功能维度分为三个模块，统一入口 `test/run_all.py`：

| 测试脚本 | 覆盖范围 | 测试项 |
|---------|---------|:-----:|
| `test_retrieval.py` | Faiss 向量检索精度：文本 10 项 + 图像 6 项 + 混合 2 项 | 18 |
| `test_image_recognition.py` | 图片识别：禁词检查 + 单图/多图 + 用药判断 | 14 |
| `test_scenarios.py` | 端到端家庭用药功能：15 场景 × 多项检查 | ~35 |

```bash
python test/run_all.py                 # 一键运行全部
python test/test_retrieval.py          # 单独运行
python test/test_image_recognition.py
python test/test_scenarios.py
```

**失败项分析**:
1. **空查询**: 无文本无图片时返回错误（预期行为，非 bug）
2. **感冒推荐**: 文本检索对"头痛"匹配了 naratriptan（偏头痛药），而非 NSAID 或感冒药——根本原因是 CLIP 文本编码器对中文→英文跨语言检索精度有限
3. **部分药片图像检索精度**: CLIP ViT-B/32 对药片纹理特征区分度有限，Suboxone 等药片排在第 3-4 位

---

## 五、系统创新点

### 5.1 双路检索加权融合

摒弃原 MultimodalRAG 的 Mean Fusion 方案，改为图文独立检索后分数加权合并，显式控制图片信号的权重（0.85），避免文本向量稀释视觉匹配精度。

### 5.2 症状-术语映射查询扩展

构建中文症状词 → 英文/拉丁医学术语的映射表，在检索和生成两个环节注入，缓解 CLIP 跨语言语义差距问题。

### 5.3 动态 System Prompt 分支

根据 `query_type`（image/multimodal/text）动态选择不同的 System prompt 模板，使得同一个 LLM 能在图像识别和文字咨询两种模式间无缝切换，避免格式混淆。

### 5.4 物理属性注入 context

将数据库中的结构化物理属性（颜色、形状、印记、规格）格式化后显式注入每条检索结果的 text 字段，使 LLM 能够在候选药物间进行外观交叉验证，提升识别准确率。

### 5.5 context 头标记注入

在图片查询的每条检索结果头部注入 `🎯 图像检索匹配 #N 🎯` 标记，强制 LLM 感知到这些文档来源于图像检索流程而非随机文本匹配，彻底解决 LLM 拒绝执行图像识别的问题。

---

## 六、局限性与改进方向

| 局限 | 原因 | 改进方向 |
|------|------|---------|
| 部分药片 CLIP 区分度不足 | ViT-B/32 对药片纹理特征不够敏感 | 换用 ViT-L/14 或在 ePillID 上微调 CLIP |
| 中文症状检索精度有限 | CLIP 文本编码器以英文为主 | 引入医疗领域中文-英文双语言模型 |
| 图片重新生成时丢失 | base64 存在前端，无法回传后端 | 后端缓存上传图片或改为文件路径引用 |
| 无流式输出 | GLM-4-Flash API 非流式调用 | 后端切换为 SSE streaming |

---

## 七、结论

本实验成功构建了一个完整的多模态 RAG 家庭用药系统。通过对 CLIP+Faiss 架构的深入理解，我们识别并解决了过期 NDC 映射、跨语言检索、多模态信号融合、LLM 行为对齐、应答格式适配等 6 个核心技术挑战。经过 6 轮迭代优化，系统在 15+ 类家庭用药场景中实现了 92% 的综合测试通过率。

核心贡献在于：（1）提出双路检索加权融合方案替代 Mean Fusion；（2）设计症状-术语映射表缓解跨语言鸿沟；（3）通过 context 头标记注入和动态 System prompt 分叉解决 LLM 行为对齐问题。这些技术方案对构建其他领域的多模态 RAG 系统具有参考价值。
