from __future__ import annotations

from datetime import datetime


def format_date(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def format_time(now: datetime) -> str:
    return now.strftime("%H:%M:%S")

