"""Hash-chained cumulative billing ledger for deployment #2."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from ternarystem.training import atomic_json_save


@dataclass(frozen=True)
class LedgerTotals:
    seconds: int
    cents: int


class BillingLedger:
    def __init__(
        self,
        path: str | Path,
        *,
        cents_per_hour: int = 54,
        planned_seconds: int = 39_934,
        planned_cents: int = 600,
        absolute_seconds: int = 46_601,
        absolute_cents: int = 700,
    ) -> None:
        self.path = Path(path)
        self.cents_per_hour = cents_per_hour
        self.planned = LedgerTotals(planned_seconds, planned_cents)
        self.absolute = LedgerTotals(absolute_seconds, absolute_cents)
        if min(cents_per_hour, planned_seconds, planned_cents, absolute_seconds, absolute_cents) <= 0:
            raise ValueError("ledger limits must be positive integers")
        if planned_seconds >= absolute_seconds or planned_cents >= absolute_cents:
            raise ValueError("planned limits must be below absolute limits")

    @classmethod
    def from_gate_policy(cls, path: str | Path, policy: dict) -> "BillingLedger":
        budget = policy.get("budget") if isinstance(policy, dict) else None
        if not isinstance(budget, dict):
            raise ValueError("gate policy budget is missing")
        required = {
            "approved_cents_per_hour",
            "planned_seconds",
            "planned_cents",
            "absolute_seconds",
            "absolute_cents",
        }
        if set(budget) != required or not all(
            isinstance(budget[key], int) for key in required
        ):
            raise ValueError("gate policy budget is invalid")
        return cls(
            path,
            cents_per_hour=budget["approved_cents_per_hour"],
            planned_seconds=budget["planned_seconds"],
            planned_cents=budget["planned_cents"],
            absolute_seconds=budget["absolute_seconds"],
            absolute_cents=budget["absolute_cents"],
        )

    def assert_matches_gate_policy(self, policy: dict) -> None:
        expected = type(self).from_gate_policy(self.path, policy)
        if (
            self.cents_per_hour != expected.cents_per_hour
            or self.planned != expected.planned
            or self.absolute != expected.absolute
        ):
            raise ValueError("billing ledger limits conflict with gate policy")

    @staticmethod
    def _hash(event_without_hash: dict) -> str:
        encoded = json.dumps(
            event_without_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("billing ledger is unreadable or corrupt") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("billing ledger schema is invalid")
        events = payload.get("events")
        if not isinstance(events, list):
            raise TypeError("billing ledger events are invalid")
        previous_hash = None
        seconds = 0
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise TypeError("billing ledger event is invalid")
            claimed_hash = event.get("event_sha256")
            unhashed = {key: value for key, value in event.items() if key != "event_sha256"}
            if claimed_hash != self._hash(unhashed) or event.get("previous_sha256") != previous_hash:
                raise ValueError("billing ledger hash chain is invalid")
            duration = event.get("duration_seconds")
            if not isinstance(duration, int) or duration < 0:
                raise ValueError("billing ledger duration is invalid")
            seconds += duration
            cents = math.ceil(seconds * self.cents_per_hour / 3600)
            if event.get("sequence") != index or event.get("cumulative_seconds") != seconds:
                raise ValueError("billing ledger cumulative seconds conflict")
            if event.get("cumulative_cents") != cents:
                raise ValueError("billing ledger cumulative cost conflicts")
            previous_hash = claimed_hash
        return events

    def has_transaction(self, transaction_id: str) -> bool:
        if not transaction_id:
            raise ValueError("billing transaction ID is required")
        matches = [
            event for event in self.read() if event.get("transaction_id") == transaction_id
        ]
        if len(matches) > 1:
            raise ValueError("billing transaction is duplicated")
        return bool(matches)

    def totals(self) -> LedgerTotals:
        events = self.read()
        if not events:
            return LedgerTotals(0, 0)
        final = events[-1]
        return LedgerTotals(final["cumulative_seconds"], final["cumulative_cents"])

    def append(
        self,
        *,
        category: str,
        duration_seconds: int,
        deployment_id: str,
        transaction_id: str | None = None,
    ) -> dict:
        if (
            not category
            or not deployment_id
            or not isinstance(duration_seconds, int)
            or (transaction_id is not None and not transaction_id)
        ):
            raise ValueError("ledger event fields are invalid")
        if duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        try:
            lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            raise ValueError("billing ledger is locked or unwritable") from error
        try:
            return self._append_locked(
                category, duration_seconds, deployment_id, transaction_id
            )
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)

    def _append_locked(
        self,
        category: str,
        duration_seconds: int,
        deployment_id: str,
        transaction_id: str | None,
    ) -> dict:
        events = self.read()
        if transaction_id is not None:
            matches = [event for event in events if event.get("transaction_id") == transaction_id]
            if len(matches) > 1:
                raise ValueError("billing transaction is duplicated")
            if matches:
                event = matches[0]
                if (
                    event["category"] != category
                    or event["duration_seconds"] != duration_seconds
                    or event["deployment_id"] != deployment_id
                ):
                    raise ValueError("billing transaction conflicts")
                return event
        previous = events[-1]["event_sha256"] if events else None
        previous_seconds = events[-1]["cumulative_seconds"] if events else 0
        cumulative_seconds = previous_seconds + duration_seconds
        cumulative_cents = math.ceil(cumulative_seconds * self.cents_per_hour / 3600)
        event = {
            "sequence": len(events),
            "deployment_id": deployment_id,
            "category": category,
            "duration_seconds": duration_seconds,
            "cumulative_seconds": cumulative_seconds,
            "cumulative_cents": cumulative_cents,
            "previous_sha256": previous,
        }
        if transaction_id is not None:
            event["transaction_id"] = transaction_id
        event["event_sha256"] = self._hash(event)
        events.append(event)
        try:
            atomic_json_save({"schema_version": 1, "events": events}, self.path)
            descriptor = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise ValueError("billing ledger is not writable") from error
        self.read()
        return event

    def assert_within_absolute(self, additional_seconds: int = 0) -> None:
        totals = self.totals()
        seconds = totals.seconds + additional_seconds
        cents = math.ceil(seconds * self.cents_per_hour / 3600)
        if seconds >= self.absolute.seconds or cents >= self.absolute.cents:
            raise ValueError("absolute deployment budget would be reached")

    def has_planned_reserve(self, required_seconds: int) -> bool:
        totals = self.totals()
        seconds = totals.seconds + required_seconds
        cents = math.ceil(seconds * self.cents_per_hour / 3600)
        return seconds < self.planned.seconds and cents < self.planned.cents
