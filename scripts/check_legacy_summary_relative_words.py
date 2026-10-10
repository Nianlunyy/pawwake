"""Scan session_cache_state for legacy relative time terms and print segment numbers/counts."""

import asyncio
import json
import re

from db import core as db_core

RELATIVE_WORDS = ["昨晚", "今早", "今天", "刚才", "明天", "前天", "后天", "最近", "目前", "最新状态"]


async def main():
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT session_id, summary FROM session_cache_state WHERE summary IS NOT NULL AND summary != ''")
        total_matched_parts = 0
        print(f"扫描对话线数: {len(rows)}")
        for r in rows:
            sid = r["session_id"]
            raw_summary = r["summary"] or ""
            parts = []
            try:
                parsed = json.loads(raw_summary)
                if isinstance(parsed, list):
                    parts = parsed
                else:
                    parts = [raw_summary]
            except Exception:
                parts = [raw_summary]

            matched_indices = []
            for idx, part in enumerate(parts):
                if any(w in part for w in RELATIVE_WORDS):
                    matched_indices.append(idx + 1)
            if matched_indices:
                total_matched_parts += len(matched_indices)
                print(f"Session {sid}: 共 {len(parts)} 段，其中第 {matched_indices} 段含有相对时间词")
        print(f"全部对话线含相对词的总段数: {total_matched_parts}")


if __name__ == "__main__":
    asyncio.run(main())
