import asyncio
import json
from pathlib import Path

from browser_use import Agent, ChatGoogle
from dotenv import load_dotenv

load_dotenv()

OUTPUT = Path("msrtc_results.json")

TASK = """
Use only the official MSRTC public reservation/timetable website:
https://npublic.msrtcors.com/reservation-home

Search publicly available timetable/reservation results for:
1. Mumbai -> Pune
2. Mumbai -> Bengaluru

Use a future date accepted by the public site.

Rules:
- Do not log in, book, pay, or enter personal information.
- Do not bypass CAPTCHA, authentication, rate limits, robots/access controls, or other protections.
- Do not use third-party timetable sites.
- Record only information visibly returned by the official MSRTC website.
- Never guess missing values.

For every displayed service, collect where available:
service/bus number, bus type/service type, origin, destination, departure time,
arrival time, duration, stop sequence/intermediate stops, distance,
boarding stop, and alighting stop.

Return STRICT JSON with source, retrieved_at, and routes containing services.
If a route has no results, return an empty services array.
"""

async def main() -> None:
    agent = Agent(task=TASK, llm=ChatGoogle(model="gemini-2.5-flash"))
    history = await agent.run(max_steps=60)
    result = history.final_result() or ""
    print("\n========== MSRTC RESULT ==========\n")
    print(result)
    OUTPUT.write_text(result, encoding="utf-8")
    print(f"\nSaved local result to {OUTPUT}")

if __name__ == "__main__":
    asyncio.run(main())
