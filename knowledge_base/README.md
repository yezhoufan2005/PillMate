# 家庭用药 MRAG 系统 - 知识库

## 文件结构

```
knowledge_base/
├── README.md           ← 本文件
├── database.json       ← 核心知识库 (928 种药物, 46.2 MB)
└── images/             ← 药片图像 (5,728 张, 按 NDC 分组)
    ├── 00093-0148-01/
    │   ├── 10.jpg
    │   └── ...
    ├── 00093-7248-06/
    └── ... (928 个文件夹)
```

## database.json 结构

```json
{
  "stats": {
    "total_ndc": 928,
    "total_images": 5728,
    "status_counts": { "success": 874, "rxnorm_miss": 0, "dailymed_miss": 54, "spl_parse_failed": 0 },
    "rag_coverage": { "has_rag_text": 894, "has_physical_properties": 874, ... }
  },
  "drugs": {
    "00093-0148-01": {
      "ndc": "00093-0148-01",
      "image_count": 7,
      "image_paths": ["fcn_mix_weight/dc_224/10.jpg", ...],
      "rxnorm": { "rxcui": "198012", "status": "OBSOLETE", "active": "NO" },
      "drug_name": "Naproxen",
      "generic_name": "naproxen",
      "brand_name": null,
      "manufacturer": "Sportpharm LLC",
      "physical_properties": [
        { "strength": "250 mg", "color": "yellow", "shape": "round", "imprint": "s & g", "scored": true },
        { "strength": "375 mg", "color": "yellow", "shape": "capsule", "imprint": "sg / 435" },
        { "strength": "500 mg", "color": "yellow", "shape": "oblong", "imprint": "s & g", "scored": true }
      ],
      "active_ingredients": [...],
      "inactive_ingredients": [...],
      "pharmacologic_class": [...],
      "pregnancy_category": null,
      "dosage_form": null,
      "route": [],
      "indications": "1 INDICATIONS AND USAGE ...",
      "warnings": "WARNING: RISK OF ...",
      "contraindications": "4 CONTRAINDICATIONS ...",
      "adverse_reactions": "6 ADVERSE REACTIONS ...",
      "drug_interactions": "7 DRUG INTERACTIONS ...",
      "clinical_pharmacology": "12 CLINICAL PHARMACOLOGY ...",
      "dosage_administration": "2 DOSAGE AND ADMINISTRATION ...",
      "overdosage": "10 OVERDOSAGE ...",
      "rag_text": "DRUG NAME: Naproxen\nGENERIC NAME: naproxen\n...\n\nINDICATIONS AND USAGE:\n...",
      "status": "success"
    }
  }
}
```

## 数据字段说明

### 图像相关
| 字段 | 类型 | 说明 |
|------|------|------|
| `image_count` | int | 该 NDC 对应图像数量 |
| `image_paths` | list[str] | 图像在原始 ePillID 数据集中的路径 |
| 实际图像文件 | - | 按 NDC 分组在 `images/{NDC}/` 下 |

### 核心药物信息
| 字段 | 类型 | 说明 |
|------|------|------|
| `drug_name` | str | 药物名称 (品牌或通用名) |
| `generic_name` | str | 通用名/活性成分名 |
| `brand_name` | str/null | 品牌名 (可能为 null) |
| `manufacturer` | str | 制造商名称 |
| `status` | str | "success" = 成功; "dailymed_miss" = 无说明书 |

### 物理属性 (药片识别核心)
`physical_properties` 数组中每项:
| 字段 | 类型 | 说明 |
|------|------|------|
| `strength` | str | 剂量 (如 "375 mg") |
| `color` | str | 颜色 (white/red/yellow/orange/peach/blue/green...) |
| `shape` | str | 形状 (round/capsule/oval/oblong/circular/triangular...) |
| `imprint` | str | 印记文字 |
| `scored` | bool | 是否有刻痕 |

### 说明书章节 (RAG 问答核心)
| 字段 | 说明 |
|------|------|
| `indications` | 适应症 |
| `dosage_administration` | 用法用量 |
| `contraindications` | 禁忌症 |
| `warnings` | 警告 (含黑框警告) |
| `adverse_reactions` | 副作用 |
| `drug_interactions` | 药物相互作用 |
| `clinical_pharmacology` | 临床药理学 |
| `overdosage` | 过量处理 |

### RAG 预拼接文本
| 字段 | 说明 |
|------|------|
| `rag_text` | 所有关键词段落拼接成的完整文本, 可直接喂入向量数据库 (ChromaDB/FAISS) |

## MRAG 系统使用方法

```python
import json

# 加载知识库
with open("knowledge_base/database.json", "r", encoding="utf-8") as f:
    kb = json.load(f)

# 通过 NDC 查询 (O(1))
drug = kb["drugs"]["00093-0148-01"]

# 获取药物名称
print(drug["drug_name"])           # "Naproxen"

# 获取药片外观 (用于图像匹配验证)
for p in drug["physical_properties"]:
    print(f"{p['color']} {p['shape']} {p['strength']} imprint={p['imprint']}")

# 获取 RAG 全文 (直接喂给向量库/LLM)
text = drug["rag_text"]

# 获取单独章节
print(drug["indications"])
print(drug["warnings"])
print(drug["contraindications"])

# 获取对应图像路径
for img in drug["image_paths"]:
    print(f"images/{drug['ndc']}/{img.split('/')[-1]}")
```

## 数据来源与构建链路

```
ePillID 数据集 (928 种 NDC, 5,728 张药片图像)
    │
    ▼
RxNorm API (过期 NDC → rxcui 药物本体 ID, 100% 映射成功)
    │
    ▼
DailyMed SPL XML (官方完整药品说明书, 94.2% 覆盖率)
    │
    ▼
knowledge_base/database.json (46.2 MB, 结构化药物知识库)
```

## 数据质量

| 指标 | 数值 |
|------|------|
| 药物总数 | 928 |
| 成功获取说明书 | 874 (94.2%) |
| 含物理属性 | 874 (100%) |
| 含颜色 | 789 (90.3%) |
| 含形状 | 663 (75.9%) |
| 含印记 | 394 (45.1%) |
| 含制造商 | 874 (100%) |
| 含适应症 | 798 (91.3%) |
| 含警告 | 783 (89.6%) |
| 含禁忌症 | 790 (90.4%) |
| 含副作用 | 792 (90.6%) |
| 含用法用量 | 774 (88.6%) |
| RAG 预拼接 | 894 |
| 图像总数 | 5,728 |
| 平均每药 | 6.2 张 |

## 注意事项

- NDC 编码来自 ePillID 数据集, 部分已过期 (OBSOLETE), 但通过 RxNorm 映射到当前药物本体
- 制造商、印记等字段因不同制造商的产品而有差异 (同一药物可能有多家工厂生产)
- `rag_text` 字段包含所有章节的拼接文本, 已清理 SPL 内部交叉引用标记
- `images/` 下每张图像文件名保留原始命名, 对应 `image_paths` 字段中的路径
