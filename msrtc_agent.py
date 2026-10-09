import asyncio
from pathlib import Path
from typing import Optional

from browser_use import Agent, ChatGoogle
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

OUTPUT = Path("msrtc_results.json")


class Service(BaseModel):
    service_number: Optional[str] = None
    bus_type: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    duration: Optional[str] = None
    distance_km: Optional[str] = None
    boarding_stop: Optional[str] = None
    alighting_stop: Optional[str] = None
    intermediate_stops: list[str] = Field(default_factory=list)


class RouteResult(BaseModel):
    origin: str
    destination: str
    services: list[Service] = Field(default_factory=list)


class MsrtcResult(BaseModel):
    source: str
    retrieved_at: str
    routes: list[RouteResult]


TASK = """
You must actually perform BOTH public searches on the official MSRTC website before finishing.
Do not finish early after opening the website.

OFFICIAL WEBSITE ONLY:
https://npublic.msrtcors.com/reservation-home

SEARCH 1: Mumbai -> Pune
SEARCH 2: Mumbai -> Bengaluru

For each search:
1. Open/use the official MSRTC public reservation/timetable search.
2. Enter the From city/stop as Mumbai and the To city/stop as Pune for search 1, then Bengaluru for search 2.
3. Choose a future date accepted by the public website.
4. Submit the public search and wait for the results page/table to load.
5. Inspect the complete displayed service list. Scroll through the results if necessary so you do not stop after the first visible rows.
6. Record only values visibly returned by MSRTC. Never guess or infer missing values.

Collect, when displayed: service/bus number, bus type/service type, origin, destination,
departure time, arrival time, duration, distance, boarding stop, alighting stop,
and intermediate/stop-sequence information.

IMPORTANT COMPLETION RULES:
- Both Mumbai -> Pune AND Mumbai -> Bengaluru searches must be attempted before you finish.
- If a route genuinely has no displayed services, return an empty services array for that route.
- Do not use Google/search-engine results or third-party timetable websites.
- Do not log in, book, pay, enter personal information, bypass CAPTCHA, bypass authentication,
  bypass rate limits, or bypass robots/access controls.
- Do not invent any bus or timetable data.
- Your final response must be ONLY the requested structured output, not a prose summary.
"""


async def main() -> None:
    agent = Agent(
        task=TASK,
        llm=ChatGoogle(model="gemini-3.5-flash-lite"),
        output_model_schema=MsrtcResult,
        max_steps=100,
        directly_open_url=True,
        use_vision=True,
    )

    history = await agent.run()
    result = history.structured_output

    print("\n========== MSRTC RESULT ==========\n")
    if result is None:
        print("ERROR: Agent did not return valid structured JSON.")
        raw = history.final_result() or ""
        print(raw)
        print("\nResult was NOT saved because it was not valid structured data.")
        return

    OUTPUT.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(result.model_dump_json(indent=2))
    print(f"\nSaved validated result to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
