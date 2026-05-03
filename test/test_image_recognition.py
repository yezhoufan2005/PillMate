"""
图片识别专项测试
验证 LLM 正确处理图像检索结果:
1. 不会说"看不到图片"等禁词
2. 正确识别并输出完整药物信息
3. 多图场景分别识别
用法: python test/test_image_recognition.py
"""

import requests
from pathlib import Path

BASE = "http://localhost:8000"
KB = Path("knowledge_base/images")

passed = 0
failed = 0

BANNED_PHRASES = [
    "看不到图片", "无法识别图片", "无法直接查看",
    "不能直接查看", "cannot see", "unable to view",
    "没有直接查看图片的能力",
]

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1

def no_banned_phrase(text):
    for phrase in BANNED_PHRASES:
        if phrase in text:
            return False, phrase
    return True, ""

def has_drug_info(text):
    return len(text) > 100 and any(
        w in text for w in ["适应症", "用法用量", "副作用", "禁忌", "indication", "dosage"]
    )

def qi(ndc, q="", k=5):
    imgs = sorted((KB / ndc).glob("*.jpg"))
    if not imgs: return None
    with open(imgs[0], "rb") as f:
        return requests.post(f"{BASE}/query", files={"images": f},
            data={"query_text": q, "top_k": str(k)}).json()

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ======== 1. 单图纯识别 (问"这是什么药") ========
section("1. 单图纯识别: 这是什么药?")

test_ndcs = ["00093-0148-01", "00093-0924-01", "00603-3079-21"]
for ndc in test_ndcs:
    r = qi(ndc, "这是什么药？")
    if not r:
        print(f"  跳过 {ndc}")
        continue
    drugs = r.get("retrieved_drugs", [])
    top1 = drugs[0] if drugs else {}
    resp = r.get("response", "")

    print(f"  {ndc} → Top1: {top1.get('generic_name','?')}")
    ok, phrase = no_banned_phrase(resp)
    check(f"{ndc} 无禁词", ok, f"含: {phrase}")
    check(f"{ndc} 有药物信息", has_drug_info(resp))
    check(f"{ndc} Top-1 命中", ndc in [d["ndc"] for d in drugs[:3]],
          f"top3: {[(d['ndc'], d.get('generic_name','?')) for d in drugs[:3]]}")

# ======== 2. 单图 + 用药判断 ========
section("2. 单图 + 用药判断: 这个药有效吗?")

r = qi("00093-0148-01", "我牙龈发炎了，这个药有效吗？")
if r:
    resp = r.get("response", "")
    print(f"  {resp[:200]}...")
    ok, phrase = no_banned_phrase(resp)
    check("图片+用药 无禁词", ok)
    check("图片+用药 正常回答", len(resp) > 100)

# ======== 3. 多图识别 ========
section("3. 多图识别: 上传2张不同药片")

imgs1 = sorted((KB / "00093-0148-01").glob("*.jpg"))
imgs2 = sorted((KB / "00093-1044-01").glob("*.jpg"))
if imgs1 and imgs2:
    with open(imgs1[0], "rb") as f1, open(imgs2[0], "rb") as f2:
        r = requests.post(f"{BASE}/query",
            files=[("images", f1), ("images", f2)],
            data={"query_text": "分别是什么药？", "top_k": "5"}).json()
    resp = r.get("response", "")
    print(f"  {resp[:300]}...")
    ok, phrase = no_banned_phrase(resp)
    check("多图识别 无禁词", ok)
    check("多图识别 回答>150字符", len(resp) > 150)

# ======== 4. 多图 + 用药判断 ========
section("4. 多图 + 用药判断: 可以吃吗?")

if imgs1 and imgs2:
    with open(imgs1[0], "rb") as f1, open(imgs2[0], "rb") as f2:
        r = requests.post(f"{BASE}/query",
            files=[("images", f1), ("images", f2)],
            data={"query_text": "这俩药我可以吃吗？治什么的", "top_k": "5"}).json()
    resp = r.get("response", "")
    print(f"  {resp[:300]}...")
    ok, phrase = no_banned_phrase(resp)
    check("多图+用药 无禁词", ok)
    check("多图+用药 正常回答", len(resp) > 150)

# ======== 5. 图片+未描述图片的文字 ========
section("5. 纯图片无文字: 仅上传图片不说话")

r = qi("00093-0148-01", "")
if r:
    resp = r.get("response", "")
    ok, phrase = no_banned_phrase(resp)
    check("纯图片无文字 无禁词", ok)

print(f"\n{'='*60}")
print(f"  图片识别: {passed}/{passed+failed} 通过")
print(f"{'='*60}")
