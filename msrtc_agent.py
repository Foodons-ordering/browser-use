import asyncio
from datetime import datetime, timezone
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
    search_completed: bool
    services: list[Service] = Field(default_factory=list)


class MsrtcResult(BaseModel):
    source: str
    retrieved_at: str
    routes: list[RouteResult]


BASE_URL = "https://npublic.msrtcors.com/reservation-home"


async def search_route(origin: str, destination: str) -> RouteResult:
    task = f"""
You are searching ONE route only: {origin} -> {destination}.

Use ONLY this official MSRTC public reservation/timetable website:
{BASE_URL}

You MUST complete the public search before calling done.

Required interaction sequence:
1. On the MSRTC page, find the From field.
2. Type exactly: {origin}
3. If an autocomplete/dropdown appears, CLICK the matching {origin} option. Typing text alone is NOT a completed step.
4. Find the To field.
5. Type exactly: {destination}
6. If an autocomplete/dropdown appears, CLICK the matching {destination} option. Typing text alone is NOT a completed step.
7. Choose a future date accepted by the public site.
8. CLICK the public search/submit button.
9. WAIT for the search result page/table to load.
10. Only after the results page is visible, inspect ALL displayed services. Scroll if needed.

CRITICAL: Do NOT call done while still on the form, while an autocomplete dropdown is open,
or immediately after typing a city. The task is unfinished until the search button has been
clicked and the resulting page/table has been inspected.

Record ONLY information visibly returned by MSRTC. Never guess.
For every displayed service collect when available:
service/bus number, bus/service type, origin, destination, departure time, arrival time,
duration, distance, boarding stop, alighting stop, and intermediate/stop-sequence information.

If the official result page explicitly shows no services, set search_completed=true and services=[].
If you cannot complete the public search because the website itself prevents it, set
search_completed=false and services=[]. Do not pretend the search was completed.

Do not use search engines or third-party timetable sites.
Do not log in, book, pay, enter personal information, or bypass CAPTCHA, authentication,
rate limits, robots/access controls, or other protections.

Return ONLY the structured RouteResult.
"""

    agent = Agent(
        task=task,
        llm=ChatGoogle(model="gemini-3.5-flash-lite"),
        output_model_schema=RouteResult,
        max_steps=80,
        max_actions_per_step=3,
        use_vision=True,
        use_judge=True,
        directly_open_url=True,
    )

    history = await agent.run()
    result = history.structured_output
    if result is None:
        raise RuntimeError(
            f"No structured result for {origin}->{destination}: "
            f"{history.final_result() or 'empty final result'}"
        )
    return result


async def main() -> None:
    routes: list[RouteResult] = []

    for origin, destination in (("Mumbai", "Pune"), ("Mumbai", "Bengaluru")):
        print(f"\n========== SEARCHING {origin} -> {destination} ==========")
        try:
            result = await search_route(origin, destination)
        except Exception as exc:
            print(f"ERROR: {exc}")
            print("This route was NOT treated as successfully searched.")
            continue

        print(result.model_dump_json(indent=2))
        routes.append(result)

    # Never save partial data as if both routes were completed.
    required = {("Mumbai", "Pune"), ("Mumbai", "Bengaluru")}
    completed = {(r.origin, r.destination) for r in routes if r.search_completed}

    if completed != required:
        print("\nERROR: Both required route searches were not completed.")
        print(f"Completed routes: {sorted(completed)}")
        print("No importable dataset was saved.")
        return

    final = MsrtcResult(
        source=BASE_URL,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        routes=routes,
    )
    OUTPUT.write_text(final.model_dump_json(indent=2), encoding="utf-8")
    print("\n========== FINAL VALIDATED RESULT ==========")
    print(final.model_dump_json(indent=2))
    print(f"\nSaved validated result to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
