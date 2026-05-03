"""
全场景功能测试
覆盖 15 类家庭用药场景，验证端到端回答质量
用法: python test/test_scenarios.py
"""

import requests
from pathlib import Path

BASE = "http://localhost:8000"
KB = Path("knowledge_base/images")

passed = 0
failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1

def qt(q, k=5):
    return requests.post(f"{BASE}/query", data={"query_text": q, "top_k": str(k)}).json()

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

def has_response(r):
    return r.get("success") and len(r.get("response", "")) > 50
def has_disclaimer(r):
    return "不构成医疗建议" in r.get("response", "")
def no_image_format(r):
    """纯文本回答不应含图片识别格式"""
    return "识别结果" not in r.get("response", "")
def has_recommend(r):
    return "推荐药物" in r.get("response", "")

# ==========================================
#  场景 1-4: 症状匹配推荐
# ==========================================
section("场景1: 发炎吃什么药?")
r = qt("发炎了吃什么药？")
print(f"  Top1: {r['retrieved_drugs'][0]['generic_name']} | {r['response'][:150]}...")
check("无'识别结果'", no_image_format(r))
check("有'推荐药物'", has_recommend(r))
check("有免责声明", has_disclaimer(r))

section("场景2: 头痛吃什么药?")
r = qt("头痛得厉害，吃什么药？")
print(f"  回答: {r['response'][:200]}...")
check("有推荐", has_recommend(r))
check("含止痛相关词", any(w in r["response"] for w in ["止", "痛", "pain", "analgesic", "headache", "缓解"]))

section("场景3: 感冒发烧咳嗽")
r = qt("感冒了，咳嗽发烧，该吃什么药？")
print(f"  Top1: {r['retrieved_drugs'][0]['generic_name']} | {r['response'][:200]}...")
check("有推荐", has_recommend(r))
check("回答>100字符", len(r["response"]) > 100)

section("场景4: 皮肤过敏痒红肿")
r = qt("皮肤过敏痒红肿吃什么药？")
print(f"  回答: {r['response'][:200]}...")
check("无'识别结果'", no_image_format(r))
check("含过敏相关", any(w in r["response"] for w in ["抗过敏", "antihistamine", "过敏", "allergy", "抗组胺"]))

# ==========================================
#  场景 5-7: 药物信息咨询
# ==========================================
section("场景5: Naproxen 副作用")
r = qt("Naproxen 萘普生有什么副作用？禁用于什么人？")
print(f"  回答: {r['response'][:200]}...")
check("无'识别结果'", no_image_format(r))
check("含副作用", "副作用" in r["response"] or "adverse" in r["response"].lower())

section("场景6: 布洛芬用法用量")
r = qt("布洛芬 Ibuprofen 的用法用量是多少？一次吃几片？")
print(f"  回答: {r['response'][:200]}...")
check("含剂量信息", any(w in r["response"] for w in ["mg", "毫克", "剂量", "dose", "片", "capsule"]))

section("场景7: 二甲双胍怎么吃")
r = qt("Metformin 二甲双胍一次吃多少？什么时候吃？")
print(f"  回答: {r['response'][:200]}...")
check("含服用指导", any(w in r["response"] for w in ["mg", "毫克", "剂量", "dose", "餐", "meal", "服用", "口服"]))

# ==========================================
#  场景 8-9: 药效判断
# ==========================================
section("场景8: 萘普生治头痛有效吗?")
r = qt("头痛，萘普生 Naproxen 有效吗？能吃吗？")
print(f"  回答: {r['response'][:200]}...")
check("无'识别结果'", no_image_format(r))
check("有药效判断", any(w in r["response"] for w in ["有效", "可以", "适用", "indicated", "缓解", "treat"]))

section("场景9: 阿托伐他汀伤肝吗?")
r = qt("Atorvastatin 阿托伐他汀伤肝吗？有什么副作用？")
print(f"  回答: {r['response'][:200]}...")
check("含肝/副作用", "副作用" in r["response"] or "肝" in r["response"] or "liver" in r["response"].lower())

# ==========================================
#  场景 10-11: 禁忌排查
# ==========================================
section("场景10: 孕妇能吃华法林吗?")
r = qt("孕妇能吃 Warfarin 华法林吗？有什么风险？")
print(f"  回答: {r['response'][:200]}...")
check("提及孕妇风险", any(w in r["response"] for w in ["孕妇", "pregnancy", "胎儿", "fetal", "怀孕"]))
check("有免责声明", has_disclaimer(r))

section("场景11: 高血压吃什么药?")
r = qt("血压高吃什么药好？")
print(f"  回答: {r['response'][:200]}...")
check("推荐降压药", any(w in r["response"] for w in ["压", "hypertension", "lisinopril", "amlodipine", "hydrochlorothiazide", "diuretic"]))
check("无'识别结果'", no_image_format(r))

# ==========================================
#  场景 12-13: 图片混合场景
# ==========================================
section("场景12: 图片 + 牙龈发炎")
r = qi("00093-0148-01", "我牙龈发炎了，这个药有效吗？能消肿吗？")
if r:
    resp = r.get("response", "")
    print(f"  回答: {resp[:250]}...")
    check("无'看不到图片'", "看不到图片" not in resp)
    check("回答>100字符", len(resp) > 100)
    check("有针对性", any(w in resp for w in ["炎症", "抗炎", "anti", "swelling", "inflammation", "肿"]))

section("场景13: 图片 + 无文字")
r = qi("00093-0148-01", "")
if r:
    check("空文字仍返回正常回答", len(r.get("response", "")) > 50)

# ==========================================
#  场景 14-15: 边界情况
# ==========================================
section("场景14: 知识库不存在的药")
r = qt("Oseltamivir 奥司他韦是什么药？")
print(f"  回答: {r['response'][:200]}...")
check("有免责声明", has_disclaimer(r))

section("场景15: 空查询")
r = qt("")
check("空查询返回错误/短提示", not r.get("success", True) or len(r.get("response", "")) < 50,
      f"返回: {r.get('response','')[:100]}")

# ==========================================
print(f"\n{'='*60}")
total = passed + failed
pct = passed * 100 // total if total > 0 else 0
print(f"  全场景测试: {passed}/{total} 通过 ({pct}%)")
print(f"{'='*60}")
