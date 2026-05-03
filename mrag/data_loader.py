"""
知识库加载适配器
将 knowledge_base/database.json + images/ 转换为 RAGCore 所需的文档格式
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import KB_JSON, KB_IMAGES

logger = logging.getLogger(__name__)


def load_drug_documents(
    json_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    max_docs: int = 0,  # 测试模式: 默认只加载 50 条; 设 0 加载全部
) -> List[Dict[str, Any]]:
    """
    从知识库加载药物文档，适配为 RAGCore 格式。

    每个文档:
    {
        "id": "00093-0148-01",           # NDC 作为文档ID
        "text": "DRUG NAME: Naproxen...", # rag_text 预拼接文本
        "image_path": ".../00093-0148-01/10.jpg",  # 第一张图像路径
        "ndc": "00093-0148-01",
        "drug_name": "Naproxen",
        "physical_properties": [...],
        "raw": { ... }                    # 完整原始数据
    }

    Returns:
        List[Dict]: 文档列表
    """
    json_path = json_path or str(KB_JSON)
    image_dir = image_dir or str(KB_IMAGES)

    with open(json_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    drugs = kb.get("drugs", {})

    documents: List[Dict[str, Any]] = []
    found_images = 0
    skipped_no_rag = 0

    for ndc, drug in drugs.items():
        rag_text = drug.get("rag_text", "").strip()
        if not rag_text:
            skipped_no_rag += 1
            continue

        image_paths = drug.get("image_paths", [])
        image_path: Optional[str] = None

        if image_paths:
            ndc_img_dir = Path(image_dir) / ndc
            for img_rel in image_paths:
                img_name = Path(img_rel).name
                full_path = ndc_img_dir / img_name
                if full_path.exists():
                    image_path = str(full_path)
                    found_images += 1
                    break

        doc = {
            "id": ndc,
            "text": rag_text,
            "image_path": image_path,
            "ndc": ndc,
            "drug_name": drug.get("drug_name"),
            "generic_name": drug.get("generic_name"),
            "brand_name": drug.get("brand_name"),
            "manufacturer": drug.get("manufacturer"),
            "physical_properties": drug.get("physical_properties", []),
            "indications": drug.get("indications"),
            "warnings": drug.get("warnings"),
            "contraindications": drug.get("contraindications"),
            "raw": drug,
        }
        documents.append(doc)

        if max_docs > 0 and len(documents) >= max_docs:
            break

    logger.info(f"Loaded {len(documents)} drugs from knowledge base")
    logger.info(f"  with images: {found_images}")
    logger.info(f"  skipped (no rag_text): {skipped_no_rag}")

    return documents
