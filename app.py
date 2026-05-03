"""
家庭用药 MRAG 系统 - FastAPI 服务入口
核心引擎: mrag/engine.py (MultimodalEncoder + Indexer + Retriever + Generator)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from mrag.engine import setup_logging, Indexer, Retriever, Generator

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from mrag.config import *
from mrag.data_loader import load_drug_documents

# --- 日志 ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
setup_logging(str(LOG_FILE))
logger = logging.getLogger("mrag")

# --- FastAPI ---
app = FastAPI(title="家庭用药 MRAG 系统", version="1.0.0")

# --- 全局 ---
indexer: Optional[Indexer] = None
retriever: Optional[Retriever] = None
generator: Optional[Generator] = None
documents: List[Dict] = []
system_ready = False


@app.on_event("startup")
def startup():
    global indexer, retriever, generator, documents, system_ready

    logger.info("=" * 60)
    logger.info("家庭用药 MRAG 系统启动中...")
    logger.info("=" * 60)

    try:
        # 1. 加载知识库文档
        logger.info("[1/3] 加载药物知识库...")
        documents = load_drug_documents()
        docs_for_indexer = [
            {"id": d["id"], "text": d["text"], "image_path": d["image_path"]}
            for d in documents
        ]
        logger.info(f"  加载了 {len(documents)} 条药物文档")

        # 2. 初始化索引器 (Indexer 内部自带 CLIP 编码器)
        logger.info("[2/3] 初始化 Faiss 索引器...")
        indexer = Indexer(
            db_path=DB_FILE,
            faiss_text_index_path=FAISS_TEXT_INDEX,
            faiss_image_index_path=FAISS_IMAGE_INDEX,
            faiss_mean_index_path=FAISS_MEAN_INDEX,
            clip_model_name=CLIP_MODEL_NAME,
        )
        count = indexer.get_document_count()
        if count > 0:
            logger.info(f"  已有索引 ({count} 条), 跳过重建")
        else:
            logger.info(f"  新建索引 ({len(docs_for_indexer)} 条), 请等待...")
            indexer.index_documents(docs_for_indexer)
            indexer.save_indices()
            logger.info(f"  索引完成!")

        # 3. 检索器 + 生成器
        logger.info("[3/3] 初始化检索器和生成器...")
        retriever = Retriever(indexer)
        generator = Generator(model_name=LLM_MODEL)

        system_ready = True
        logger.info("=" * 60)
        logger.info(f"系统就绪! 共 {len(documents)} 种药物已索引")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"启动失败: {e}", exc_info=True)
        system_ready = False


@app.get("/health")
def health():
    return {
        "status": "ready" if system_ready else "initializing",
        "documents": len(documents),
    }


@app.post("/query")
async def query(
    query_text: str = Form(""),
    images: List[UploadFile] = File(None),
    top_k: int = Form(5),
):
    if not system_ready:
        return JSONResponse({"error": "系统尚未就绪, 请等待索引完成"}, status_code=503)

    try:
        image_paths: List[str] = []
        if images:
            UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            for img in images:
                if img and img.filename:
                    safe_name = f"upload_{int(time.time()*1000)}_{Path(img.filename).suffix}"
                    img_path = str(UPLOAD_TEMP_DIR / safe_name)
                    with open(img_path, "wb") as f:
                        f.write(await img.read())
                    image_paths.append(img_path)

        # 构建 Retriever 查询 + 检索
        has_text = bool(query_text.strip())
        has_image = len(image_paths) > 0

        if has_text and has_image:
            query_type = "multimodal"
            # 多图检索：对每张图独立检索，取最高相似度 max-pooling
            all_img_map = {}
            for img_path in image_paths:
                img_results = retriever.retrieve({"image_path": img_path}, k=top_k)
                for r in img_results:
                    did = r.get("id") or r.get("doc_id", "")
                    s = r.get("score", 0)
                    if s > all_img_map.get(did, {}).get("score", -1):
                        all_img_map[did] = dict(r)
                        all_img_map[did]["score"] = s

            txt_results = retriever.retrieve(query_text, k=top_k)
            merged = {}
            for did, r in all_img_map.items():
                merged[did] = r.get("score", 0) * 0.85
            for r in txt_results:
                did = r.get("id") or r.get("doc_id", "")
                merged[did] = merged.get(did, 0) + r.get("score", 0) * 0.15
            sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
            results = []
            for doc_id, score in sorted_items:
                r = all_img_map.get(doc_id)
                if r:
                    r_copy = dict(r)
                    r_copy["score"] = score
                    results.append(r_copy)
            results = results[:top_k]
        elif has_image:
            query_type = "image"
            all_img_map = {}
            for img_path in image_paths:
                img_results = retriever.retrieve({"image_path": img_path}, k=8)
                for r in img_results:
                    did = r.get("id") or r.get("doc_id", "")
                    s = r.get("score", 0)
                    if s > all_img_map.get(did, {}).get("score", -1):
                        all_img_map[did] = dict(r)
                        all_img_map[did]["score"] = s
            sorted_items = sorted(all_img_map.values(), key=lambda x: x.get("score", 0), reverse=True)
            results = sorted_items[:8]
        else:
            query_type = "text"
            results = retriever.retrieve(query_text, k=top_k)

        # 组装上下文 + 富集药物元数据
        context_docs = []
        retrieved_drugs = []

        for i, r in enumerate(results):
            doc_id = r.get("id") or r.get("doc_id", "")
            score = r.get("score", 0)

            matched = None
            for d in documents:
                if d["id"] == doc_id:
                    matched = d
                    break

            raw_text = matched["text"][:4000] if matched else r.get("text", "")
            phys = matched.get("physical_properties", []) if matched else []
            pp_lines = []
            for pp in phys:
                parts = []
                if pp.get("color"): parts.append(f"颜色: {pp['color']}")
                if pp.get("shape"): parts.append(f"形状: {pp['shape']}")
                if pp.get("imprint"): parts.append(f"印记: {pp['imprint']}")
                if pp.get("strength"): parts.append(f"规格: {pp['strength']}")
                if parts: pp_lines.append(" | ".join(parts))
            if pp_lines:
                raw_text = "--- 药片外观特征 ---\n" + "\n".join(pp_lines) + "\n\n" + raw_text

            if query_type in ("image", "multimodal"):
                rank_marker = f"🎯 图像检索匹配 #{i+1}（系统通过用户上传的药片图片视觉匹配到此药物）🎯"
                raw_text = rank_marker + "\n\n" + raw_text

            context_docs.append({
                "id": doc_id,
                "text": raw_text,
                "score": score,
                "drug_name": matched.get("drug_name") if matched else None,
                "physical_properties": phys,
            })
            if matched:
                retrieved_drugs.append({
                    "ndc": matched["ndc"],
                    "drug_name": matched.get("drug_name"),
                    "generic_name": matched.get("generic_name"),
                    "manufacturer": matched.get("manufacturer"),
                    "physical_properties": matched.get("physical_properties", []),
                    "similarity": float(score) if score else None,
                })

        # 生成回答 - 根据查询类型和内容智能注入引导
        if query_type in ("image", "multimodal"):
            num_images = len(image_paths)
            if num_images > 1:
                user_query = (
                    f"用户上传了{num_images}张不同的药片图片。"
                    "请分别识别每种药物并给出完整信息。"
                )
            else:
                user_query = "用户上传了一张药片图片，请识别并给出完整信息。"
            if query_text:
                user_query += f"\n\n用户额外问题: {query_text}"
        elif query_text:
            # 纯文本查询：检测症状类关键词，注入智能引导
            symptom_map = {
                "发炎": "消炎;抗炎;抗炎药;NSAID;inflammation;anti-inflammatory",
                "炎症": "消炎;抗炎;抗炎药;NSAID;inflammation;anti-inflammatory",
                "感冒": "cold;flu;influenza;decongestant;antihistamine;咳嗽;发烧;fever;cough",
                "咳嗽": "cough;cold;antitussive;expectorant",
                "发烧": "fever;antipyretic;退烧;发热;temperature",
                "头痛": "headache;pain;migraine;migraine;analgesic;偏头痛",
                "牙痛": "toothache;dental;pain;analgesic;牙;tooth",
                "胃": "stomach;gastric;ulcer;antacid;消化不良;acid",
                "腹泻": "diarrhea;antidiarrheal;gastrointestinal;肠;intestinal",
                "过敏": "allergy;antihistamine;allergic;rash;hives;urticaria",
                "失眠": "insomnia;sleep;sedative;hypnotic;睡眠",
                "抑郁": "depression;antidepressant;SSRI;mood;mood",
                "焦虑": "anxiety;anxiolytic;anxiety;nervous;panic",
                "血压": "hypertension;blood pressure;antihypertensive;high blood",
                "血糖": "diabetes;glucose;insulin;antidiabetic;糖尿病",
                "疼": "pain;analgesic;pain reliever;NSAID;止痛",
                "肿": "swelling;inflammation;anti-inflammatory;edema",
                "出血": "bleeding;hemorrhage;anticoagulant;coagulation",
                "痒": "itching;antihistamine;rash;allergy;antipruritic",
                "吐": "nausea;vomiting;antiemetic",
            }
            symptom_words = []
            for kw, expansion in symptom_map.items():
                if kw in query_text:
                    symptom_words.append(kw)
            is_symptom_query = len(symptom_words) > 0

            if is_symptom_query:
                expanded = "; ".join(symptom_map[w] for w in symptom_words)
                user_query = (
                    f"用户描述了以下症状: {query_text}\n"
                    f"关键症状: {', '.join(symptom_words)}\n"
                    f"症状对应的医学术语: {expanded}\n\n"
                    "请根据参考文档中药物的适应症说明，选出适应症与用户症状最匹配的1-2种药物。"
                    "仔细对照每个候选药物的'INDICATIONS AND USAGE'字段，只推荐适应症明确覆盖用户症状的药物。"
                    "如果所有候选药物的适应症都不匹配用户症状，请诚实告知。"
                )
            else:
                user_query = query_text
        else:
            user_query = query_text

        response = generator.generate(user_query, context_docs, query_type=query_type)

        # 保存查询记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = QUERY_RESULTS_DIR / f"{timestamp}_{query_type}"
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(session_dir / "query.json", "w", encoding="utf-8") as f:
            json.dump({
                "query_text": query_text,
                "query_type": query_type,
                "top_k": top_k,
                "retrieved_drugs": retrieved_drugs,
                "response": response,
            }, f, ensure_ascii=False, indent=2)

        for ip in image_paths:
            if Path(ip).exists():
                Path(ip).unlink()

        return {
            "success": True,
            "query_type": query_type,
            "response": response,
            "retrieved_drugs": retrieved_drugs,
        }

    except Exception as e:
        logger.error(f"查询失败: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/rebuild_index")
def rebuild_index():
    global indexer, retriever, documents
    if not system_ready:
        return JSONResponse({"error": "系统尚未就绪"}, status_code=503)

    try:
        for f in [FAISS_TEXT_INDEX, FAISS_IMAGE_INDEX, FAISS_MEAN_INDEX, DB_FILE]:
            p = Path(f)
            if p.exists():
                p.unlink()

        indexer = Indexer(
            db_path=DB_FILE,
            faiss_text_index_path=FAISS_TEXT_INDEX,
            faiss_image_index_path=FAISS_IMAGE_INDEX,
            faiss_mean_index_path=FAISS_MEAN_INDEX,
            clip_model_name=CLIP_MODEL_NAME,
        )
        docs = [{"id": d["id"], "text": d["text"], "image_path": d["image_path"]} for d in documents]
        indexer.index_documents(docs)
        indexer.save_indices()
        retriever = Retriever(indexer)

        return {"success": True, "count": len(docs)}
    except Exception as e:
        logger.error(f"重建失败: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# --- 静态前端 ---
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    logger.info(f"启动服务 http://{SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
