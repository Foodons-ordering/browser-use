import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from browser_use import Agent, ChatGoogle
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

OUTPUT = Path("msrtc_results.json")
BASE_URL = "https://npublic.msrtcors.com/reservation-home"


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
    search_completed: bool
    services: list[Service] = Field(default_factory=list)


class MsrtcResult(BaseModel):
    source: str
    retrieved_at: str
    routes: list[RouteResult]


TASK = """
Collect the CURRENT publicly displayed MSRTC services for ONE route only:
MUMBAI -> PUNE.

Use ONLY the official MSRTC public reservation/timetable website:
https://npublic.msrtcors.com/reservation-home

Do not search Bengaluru or any other route.
Do not use Google/search engines or third-party timetable websites.
Do not log in, book, pay, enter personal information, or bypass CAPTCHA,
authentication, rate limits, robots/access controls, or any other protection.

You MUST actually complete the public MSRTC search before calling done.

INTERACTION SEQUENCE:
1. Open the official MSRTC reservation/timetable page.
2. Locate the From field.
3. Type "Mumbai".
4. Wait for the autocomplete/dropdown.
5. CLICK the matching Mumbai option. Typing alone is not sufficient.
6. Locate the To field.
7. Type "Pune".
8. Wait for the autocomplete/dropdown.
9. CLICK the matching Pune option. Typing alone is not sufficient.
10. Choose a future date accepted by the website.
11. CLICK the public Search/Submit button.
12. WAIT until the results page/table is actually visible.
13. Inspect the COMPLETE displayed service list. Scroll through all result rows/pages that are
part of the public result so you don't stop at the first visible services.

CRITICAL:
- Never call done while still on the search form.
- Never call done merely because Mumbai/Pune text was typed.
- Never claim search_completed=true unless the Search button was clicked and a result page/table
was actually displayed.
- If the official result explicitly says there are no services, set search_completed=true and
services=[].
- If the website prevents completion of the public search, set search_completed=false and
services=[]. Do not invent data.

For every service visibly returned by MSRTC, collect when available:
service/bus number, bus/service type, origin, destination, departure time, arrival time,
duration, distance, boarding stop, alighting stop, and intermediate/stop-sequence information.
Record only values visible on the official MSRTC site. Never guess missing values.

Return ONLY the structured RouteResult.
"""


async def main() -> None:
    agent = Agent(
        task=TASK,
        llm=ChatGoogle(model="gemini-3.5-flash-lite"),
        output_model_schema=RouteResult,
        max_steps=100,
        max_actions_per_step=2,
        use_vision=True,
        use_judge=True,
        directly_open_url=True,
    )

    history = await agent.run()
    result = history.structured_output

    print("\n========== MSRTC MUMBAI -> PUNE ==========")
    if result is None:
        print("ERROR: No structured result returned.")
        print(history.final_result() or "empty final result")
        print("No importable dataset was saved.")
        return

    print(result.model_dump_json(indent=2))

    if not result.search_completed:
        print("\nERROR: Official Mumbai -> Pune search was not actually completed.")
        print("No importable dataset was saved.")
        return

    final = MsrtcResult(
        source=BASE_URL,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        routes=[result],
    )
    OUTPUT.write_text(final.model_dump_json(indent=2), encoding="utf-8")

    print("\n========== VALIDATED RESULT ==========")
    print(f"Mumbai -> Pune services found: {len(result.services)}")
    print(f"Saved validated result to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
