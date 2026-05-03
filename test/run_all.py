"""
PillMate 测试套件入口
依次运行检索精度 → 图片识别 → 全场景功能, 生成汇总报告
用法: python test/run_all.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

TEST_DIR = Path(__file__).parent

TESTS = [
    ("test_retrieval.py",        "检索精度测试"),
    ("test_image_recognition.py", "图片识别测试"),
    ("test_scenarios.py",         "全场景功能测试"),
]

def main():
    print("=" * 60)
    print(f"  PillMate 测试套件")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_passed = True
    results = []

    for script, desc in TESTS:
        path = TEST_DIR / script
        if not path.exists():
            print(f"\n  SKIP: {script} (文件不存在)")
            results.append((desc, "SKIPPED"))
            continue

        print(f"\n{'─'*60}")
        print(f"  [{desc}] 运行中...")
        print(f"{'─'*60}")

        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(TEST_DIR.parent),
            capture_output=True,
            text=True,
            timeout=300,
        )

        output = proc.stdout + proc.stderr
        print(output[-1500:])  # Show tail of output

        if proc.returncode != 0:
            print(f"  >>> FAILED (exit code {proc.returncode})")
            all_passed = False
            results.append((desc, "FAILED"))
        else:
            results.append((desc, "OK"))

    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)
    for desc, status in results:
        mark = "✅" if status == "OK" else ("❌" if status == "FAILED" else "⏭️")
        print(f"  {mark} {desc}")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
