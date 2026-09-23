import re
import time
from pathlib import Path

from blog_draft_agent.graph import graph

print("노드:", list(graph.get_graph().nodes))

run_started = time.strftime("%H:%M:%S")
run_t0 = time.perf_counter()

result = graph.invoke({
    "meta": {},
    "post_type": "",
    "campaign_template": {},
    "posting": "",
    "feedback": "",
    "grade": "",
    "tries": 0,
})

run_elapsed = time.perf_counter() - run_t0
run_ended = time.strftime("%H:%M:%S")
print(f"[전체 실행] 시작 {run_started} -> 종료 {run_ended} ({run_elapsed:.2f}초)")


def next_output_path(root: Path = Path(".")) -> Path:
    """output.md, output2.md, ... 중 가장 큰 번호 다음 파일 경로를 돌려준다."""
    numbers = [0]
    for p in root.glob("output*.md"):
        m = re.fullmatch(r"output(\d*)\.md", p.name)
        if m:
            numbers.append(int(m.group(1)) if m.group(1) else 1)
    next_n = max(numbers) + 1
    return Path(f"output{next_n}.md")


out_path = next_output_path()
out_path.write_text(result["posting"], encoding="utf-8-sig")
print("저장 위치:", out_path)
