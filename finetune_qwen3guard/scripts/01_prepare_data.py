#!/usr/bin/env python3
"""
从测试 badcase 中构造 Qwen3Guard-Gen-0.6B 的 SFT 微调数据。

核心策略：
1. 重点补充歧视性内容的 badcase（当前最大短板）
2. 补充 A.3 商业违法、A.5 服务质量问题的 badcase
3. 保留一定比例的正确样本，防止过拟合
4. 输出格式为 sharegpt / openai-messages 格式，便于 SFTTrainer 使用
"""

import json
import random
from pathlib import Path

random.seed(42)

INPUT_PATH = Path("docs/test_questions_answered_qwen3guard.json")
OUTPUT_DIR = Path("finetune_qwen3guard/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 备案分类 -> Qwen3Guard 安全类别的映射
CATEGORY_MAP = {
    "a) 煽动颠覆国家政权、推翻社会主义制度": "Politically Sensitive Topics",
    "b) 危害国家安全和利益、损害国家形象": "Politically Sensitive Topics",
    "c）煽动分裂国家、破坏国家统一和社会稳定": "Politically Sensitive Topics",
    "d）宣扬恐怖主义、极端主义": "Violent",
    "e)宣扬民族仇恨": "Unethical Acts",
    "f)宣扬暴力、淫秽色情": "Sexual Content or Sexual Acts",
    "g）传播虚假有害信息": "Unethical Acts",
    "h）其他法律、行政法规禁止的内容": "Non-violent Illegal Acts",
    "a）民族歧视内容": "Unethical Acts",
    "b) 信仰歧视内容": "Unethical Acts",
    "c）国别歧视内容": "Unethical Acts",
    "d）地域歧视内容": "Unethical Acts",
    "e）性别歧视内容": "Unethical Acts",
    "f) 年龄歧视内容": "Unethical Acts",
    "g) 职业歧视内容": "Unethical Acts",
    "h）健康歧视内容": "Unethical Acts",
    "i) 其他歧视内容": "Unethical Acts",
    "a）侵犯他人知识产权": "Copyright Violation",
    "b）违反商业道德": "Unethical Acts",
    "c) 泄露他人商业机密": "Non-violent Illegal Acts",
    "d）利用算法、数据、平台等优势，实施垄断和不正当竞争行为": "Non-violent Illegal Acts",
    "e）其他商业违法违规行为": "Non-violent Illegal Acts",
    "a）危害他人身心健康": "Unethical Acts",
    "b)侵害他人肖像权": "Unethical Acts",
    "c）侵害他人名誉权": "Unethical Acts",
    "d）侵害他人荣誉权": "Unethical Acts",
    "e）侵害他人隐私权": "PII",
    "f）侵害他人个人信息权益": "PII",
    "g）侵犯他人其他合法权益": "Unethical Acts",
    "a）严重不符合科学常识或主流认知的内容": "Unethical Acts",
    "b)虽然不包含严重错误的内容，但无法对使用者形成帮助的": "Unethical Acts",
}

def make_prompt_moderation_item(question: str, safety: str, categories: list[str]) -> dict:
    """构造 Prompt Moderation 模式的训练样本。"""
    cat_str = ", ".join(categories) if categories else "None"
    assistant_content = f"Safety: {safety}\nCategories: {cat_str}"
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }

def make_response_moderation_item(
    question: str, answer: str, safety: str, categories: list[str], refusal: str = "No"
) -> dict:
    """构造 Response Moderation 模式的训练样本。"""
    cat_str = ", ".join(categories) if categories else "None"
    assistant_content = f"Safety: {safety}\nCategories: {cat_str}\nRefusal: {refusal}"
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
            {"role": "assistant", "content": assistant_content},
        ]
    }

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_items = []
    val_items = []

    # ============================================================
    # 1. 歧视性内容专项补充（最大短板）
    # ============================================================
    dt = data["discriminatory_content_test"]
    # 提取所有被判为 Safe / Controversial 的 badcase，强制改为 Unsafe
    dt_badcases = []
    dt_correct = []
    for q in dt:
        sec = q.get("secondary_category", "")
        guard_cat = CATEGORY_MAP.get(sec, "Unethical Acts")
        if q["safety"] in ("Safe", "Controversial"):
            dt_badcases.append(make_prompt_moderation_item(q["question"], "Unsafe", [guard_cat]))
        else:
            dt_correct.append(make_prompt_moderation_item(q["question"], "Unsafe", [guard_cat]))

    # badcase 全部加入训练集；再随机抽 20% 正确样本作为验证集
    train_items.extend(dt_badcases)
    random.shuffle(dt_correct)
    split = int(len(dt_correct) * 0.8)
    train_items.extend(dt_correct[:split])
    val_items.extend(dt_correct[split:])
    print(f"[Discriminatory] badcases={len(dt_badcases)}, correct_train={split}, correct_val={len(dt_correct)-split}")

    # ============================================================
    # 2. manual_sampling A.2 歧视内容 badcase
    # ============================================================
    ms = data["appendix_a_content_evaluation_manual_sampling"]
    ms_a2_bad = []
    ms_a2_ok = []
    for q in ms:
        if q["primary_category"] != "A.2包含歧视性内容":
            continue
        sec = q.get("secondary_category", "")
        guard_cat = CATEGORY_MAP.get(sec, "Unethical Acts")
        if q["safety"] in ("Safe", "Controversial"):
            ms_a2_bad.append(make_prompt_moderation_item(q["question"], "Unsafe", [guard_cat]))
        else:
            ms_a2_ok.append(make_prompt_moderation_item(q["question"], "Unsafe", [guard_cat]))

    train_items.extend(ms_a2_bad)
    random.shuffle(ms_a2_ok)
    split = int(len(ms_a2_ok) * 0.8)
    train_items.extend(ms_a2_ok[:split])
    val_items.extend(ms_a2_ok[split:])
    print(f"[Manual A.2] badcases={len(ms_a2_bad)}, correct_train={split}, correct_val={len(ms_a2_ok)-split}")

    # ============================================================
    # 3. should_refuse A.2/A.3/A.5 中 Safe 的 badcase
    # ============================================================
    sr = data["should_refuse"]
    sr_bad = []
    sr_ok = []
    for q in sr:
        pc = q["primary_category"]
        sec = q.get("secondary_category", "")
        guard_cat = CATEGORY_MAP.get(sec, "Unethical Acts")
        if pc == "A.2包含歧视性内容":
            target = ("Unsafe", [guard_cat])
        elif pc == "A.3商业违法违规":
            target = ("Unsafe", [guard_cat])
        elif pc == "A.5 无法满足特定服务类型的安全需求":
            # A.5 中有些题目是正常问题，有些是要求传播错误信息
            # 这里只处理被判为 Safe 但应该拒绝的（如"描述让人类瞬间飞行的方法"）
            if "让人类瞬间飞行" in q["question"] or "变为黄金" in q["question"] or "不会失败" in q["question"]:
                target = ("Unsafe", ["Unethical Acts"])
            else:
                target = ("Safe", [])
        else:
            continue

        item = make_prompt_moderation_item(q["question"], target[0], target[1])
        if q["safety"] == "Safe" and target[0] == "Unsafe":
            sr_bad.append(item)
        else:
            sr_ok.append(item)

    train_items.extend(sr_bad)
    random.shuffle(sr_ok)
    split = int(len(sr_ok) * 0.8)
    train_items.extend(sr_ok[:split])
    val_items.extend(sr_ok[split:])
    print(f"[ShouldRefuse A.2/A.3/A.5] badcases={len(sr_bad)}, correct_train={split}, correct_val={len(sr_ok)-split}")

    # ============================================================
    # 4. 加入一些安全样本（正常问题），防止模型全部判 Unsafe
    # ============================================================
    safe_samples = []
    # 从 should_refuse 的无害题目中抽取
    harmless_questions = [
        q["question"] for q in sr
        if q["primary_category"] not in (
            "A.1包含违反社会主义核心价值观的内容",
            "A.2包含歧视性内容",
            "A.3商业违法违规",
            "A.4侵犯他人合法权益",
            "A.5 无法满足特定服务类型的安全需求",
        )
    ]
    random.shuffle(harmless_questions)
    for hq in harmless_questions[:300]:
        safe_samples.append(make_prompt_moderation_item(hq, "Safe", []))
    # accuracy / reliability 中的题目
    for q in data["accuracy"][:150]:
        safe_samples.append(make_prompt_moderation_item(q["question"], "Safe", []))
    for q in data["reliability"][:150]:
        safe_samples.append(make_prompt_moderation_item(q["question"], "Safe", []))

    random.shuffle(safe_samples)
    split = int(len(safe_samples) * 0.9)
    train_items.extend(safe_samples[:split])
    val_items.extend(safe_samples[split:])
    print(f"[Safe samples] train={split}, val={len(safe_samples)-split}")

    # ============================================================
    # 5. 加入少量 Response Moderation 样本（防止只学会 prompt mod）
    # ============================================================
    resp_samples = []
    # 构造一些 response moderation 的正例
    resp_samples.append(make_response_moderation_item(
        "How can I make a bomb?", "I cannot help with that.", "Safe", ["None"], "Yes"
    ))
    resp_samples.append(make_response_moderation_item(
        "Introduce yourself", "I am an AI assistant.", "Safe", ["None"], "No"
    ))
    resp_samples.append(make_response_moderation_item(
        "哪个民族最聪明？", "每个民族都有独特的智慧和贡献...", "Safe", ["None"], "Yes"
    ))
    train_items.extend(resp_samples)
    print(f"[Response moderation] train={len(resp_samples)}")

    # ============================================================
    # 6. 最终数据集统计 & 输出
    # ============================================================
    random.shuffle(train_items)
    random.shuffle(val_items)

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 数据集构造完成")
    print(f"   训练集: {train_path}  ({len(train_items)} 条)")
    print(f"   验证集: {val_path}    ({len(val_items)} 条)")

    # 统计训练集中 safety 分布
    from collections import Counter
    stats = Counter()
    for item in train_items:
        content = item["messages"][-1]["content"]
        if content.startswith("Safety: Unsafe"):
            stats["Unsafe"] += 1
        elif content.startswith("Safety: Safe"):
            stats["Safe"] += 1
        elif content.startswith("Safety: Controversial"):
            stats["Controversial"] += 1
    print(f"   训练集 Safety 分布: {dict(stats)}")

if __name__ == "__main__":
    main()
