import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from blog_draft_agent.config import MAX_WORKERS, SKIP_SUFFIXES, vision_client


def describe_photo(path: Path) -> str:
    """사진 한 장에서 보이는 것들을 짧은 사실 나열로 뽑는다 (완성된 문장이 아니다)."""

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    mime = "image/gif" if path.suffix.lower() == ".gif" else "image/jpeg"
    res = vision_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "이 사진에 보이는 것만 짧은 명사구로 쉼표로 나열해줘. "
                    "완성된 문장('~였습니다', '~보였어요' 등)으로 쓰지 마. "
                    "느낌이나 평가는 넣지 말고 눈에 보이는 사실만. "
                    "예: '빨간 의자, 창가 자리, 대형 스크린에 공연 영상'"
                )},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            ],
        }],
    )
    return res.choices[0].message.content.strip()


def describe_photos(photos: list[Path]) -> list[str]:
    """사진 여러 장을 동시에(병렬로) 설명한다."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(describe_photo, photos))
    return results


def describe_section(folder: Path) -> str:
    """폴더 안 사진들을 전부 설명해 이어 붙인다. 동영상·하위 폴더는 건너뛴다."""
    photos = [p for p in sorted(folder.iterdir())
              if not p.is_dir() and not p.name.startswith(".") and p.suffix.lower() not in SKIP_SUFFIXES]
    return "\n".join(describe_photos(photos))


def describe_menu(folder: Path) -> str:
    """메뉴 폴더: 요리명 하위 폴더는 요리별로 묶어 설명하고, 가격 정보(meta.txt)를 같이 붙인다."""
    price_file = folder / "meta.txt"
    prices = price_file.read_text(encoding="utf-8") if price_file.exists() else "(가격 정보 없음)"

    dish_blocks = []
    for dish_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        photos = [p for p in sorted(dish_dir.iterdir())
                  if p.suffix.lower() not in SKIP_SUFFIXES and not p.name.startswith(".")]
        descs = "\n".join(describe_photos(photos))
        dish_blocks.append(f"[{dish_dir.name}]\n{descs}")

    return f"=== 가격 정보 ===\n{prices}\n\n=== 요리별 사진 설명 ===\n" + "\n\n".join(dish_blocks)
