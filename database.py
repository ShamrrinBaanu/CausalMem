"""SQLite is the durable source of truth for custom state and transitions."""
import json, sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from config import settings
from models import State, Transition

QUARANTINE_STATUS = "quarantined"


def _is_quarantined(metadata):
    return isinstance(metadata, dict) and metadata.get("review_status") == QUARANTINE_STATUS


class Database:
    def __init__(self,path=settings.database_path):
        Path(path).parent.mkdir(parents=True,exist_ok=True); self.path=path
        with self.connect() as c:
            c.executescript("CREATE TABLE IF NOT EXISTS states(state_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,state_type TEXT,state_value TEXT,status TEXT,valid_from TEXT,valid_until TEXT,confidence REAL,source_memory_ids TEXT,metadata TEXT); CREATE INDEX IF NOT EXISTS states_user_time ON states(user_id,valid_from); CREATE TABLE IF NOT EXISTS transitions(transition_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,from_state_id TEXT,to_state_id TEXT,transition_type TEXT,relation_kind TEXT,trigger TEXT,reason TEXT,timestamp TEXT,confidence REAL,source_memory_ids TEXT,metadata TEXT); CREATE INDEX IF NOT EXISTS transitions_user_time ON transitions(user_id,timestamp);")
    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.path); c.row_factory=sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()
    def save_state(self,s,connection=None):
        d=s.model_dump(mode="json")
        values=(d["state_id"],d["user_id"],d["state_type"],d["state_value"],d["status"],d["valid_from"],d["valid_until"],d["confidence"],json.dumps(d["source_memory_ids"]),json.dumps(d["metadata"]))
        if connection is not None: connection.execute("INSERT OR REPLACE INTO states VALUES(?,?,?,?,?,?,?,?,?,?)",values)
        else:
            with self.connect() as c: c.execute("INSERT OR REPLACE INTO states VALUES(?,?,?,?,?,?,?,?,?,?)",values)
    def save_transition(self,t,connection=None):
        d=t.model_dump(mode="json")
        values=(d["transition_id"],d["user_id"],d["from_state_id"],d["to_state_id"],d["transition_type"],d["relation_kind"],d["trigger"],d["reason"],d["timestamp"],d["confidence"],json.dumps(d["source_memory_ids"]),json.dumps(d["metadata"]))
        def persist(c):
            endpoints=c.execute("SELECT state_id,user_id FROM states WHERE state_id IN (?,?)",(d["from_state_id"],d["to_state_id"])).fetchall()
            owners={row["state_id"]:row["user_id"] for row in endpoints}
            if (d["from_state_id"] not in owners or d["to_state_id"] not in owners
                    or owners[d["from_state_id"]] != d["user_id"]
                    or owners[d["to_state_id"]] != d["user_id"]):
                raise ValueError("Transition endpoints must both exist and belong to the transition user")
            c.execute("INSERT OR REPLACE INTO transitions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",values)
        if connection is not None: persist(connection)
        else:
            with self.connect() as c: persist(c)
    def states(self,user_id,include_inactive=True,include_quarantined=False):
        q="SELECT * FROM states WHERE user_id=?"+("" if include_inactive else " AND status='ACTIVE'")+" ORDER BY valid_from"
        with self.connect() as c: rows=c.execute(q,(user_id,)).fetchall()
        records=[State(state_id=x["state_id"],user_id=x["user_id"],state_type=x["state_type"],state_value=x["state_value"],status=x["status"],valid_from=x["valid_from"],valid_until=x["valid_until"],confidence=x["confidence"],source_memory_ids=json.loads(x["source_memory_ids"]),metadata=json.loads(x["metadata"] or "{}")) for x in rows]
        return [state for state in records
                if include_quarantined or not _is_quarantined(state.metadata)]
    def states_at(self,user_id,requested_timestamp,include_quarantined=False):
        """Return states valid at an aware timestamp, preserving microseconds.

        Request timestamps must be timezone-aware and are normalized to UTC.
        Stored ISO timestamps are parsed and compared as Python datetimes because
        SQLite julianday rounds fractional seconds. Legacy naive stored timestamps
        are interpreted as UTC, matching SQLite's previous behavior. NULL or
        malformed valid_from values never match; NULL valid_until is open-ended,
        while a malformed non-NULL valid_until is excluded.
        """
        if isinstance(requested_timestamp,str):
            try: requested_timestamp=datetime.fromisoformat(requested_timestamp.replace("Z","+00:00"))
            except ValueError as exc: raise ValueError("requested_timestamp must be ISO-8601") from exc
        if not isinstance(requested_timestamp,datetime):
            raise TypeError("requested_timestamp must be an aware datetime or ISO-8601 string")
        if requested_timestamp.tzinfo is None or requested_timestamp.utcoffset() is None:
            raise ValueError("requested_timestamp must include a timezone")
        instant=requested_timestamp.astimezone(timezone.utc)
        with self.connect() as c:
            # Keep the SQL user filter, then parse only this user's timestamps.
            rows=c.execute("SELECT * FROM states WHERE user_id=?",(user_id,)).fetchall()
        def stored_instant(value):
            if value is None:
                return None
            try:
                parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    parsed=parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except (AttributeError,TypeError,ValueError,OverflowError):
                return None
        matched=[]
        for row in rows:
            metadata=json.loads(row["metadata"] or "{}")
            if not include_quarantined and _is_quarantined(metadata):
                continue
            valid_from=stored_instant(row["valid_from"])
            valid_until=stored_instant(row["valid_until"])
            if valid_from is None or (row["valid_until"] is not None and valid_until is None):
                continue
            if valid_from <= instant and (valid_until is None or instant < valid_until):
                matched.append((valid_from,row,metadata))
        matched.sort(key=lambda item:(item[0],item[1]["state_id"]))
        return [State(state_id=x["state_id"],user_id=x["user_id"],state_type=x["state_type"],state_value=x["state_value"],status=x["status"],valid_from=x["valid_from"],valid_until=x["valid_until"],confidence=x["confidence"],source_memory_ids=json.loads(x["source_memory_ids"]),metadata=metadata) for _,x,metadata in matched]
    def transitions(self,user_id,include_quarantined=False):
        with self.connect() as c: rows=c.execute("SELECT * FROM transitions WHERE user_id=? ORDER BY timestamp",(user_id,)).fetchall()
        records=[Transition(transition_id=x["transition_id"],user_id=x["user_id"],from_state_id=x["from_state_id"],to_state_id=x["to_state_id"],transition_type=x["transition_type"],relation_kind=x["relation_kind"],trigger=x["trigger"],reason=x["reason"],timestamp=x["timestamp"],confidence=x["confidence"],source_memory_ids=json.loads(x["source_memory_ids"]),metadata=json.loads(x["metadata"] or "{}")) for x in rows]
        return [transition for transition in records
                if include_quarantined or not _is_quarantined(transition.metadata)]
    def close_state(self,state_id,when,connection=None):
        if connection is not None:
            connection.execute("UPDATE states SET status='INACTIVE',valid_until=? WHERE state_id=?",(when,state_id))
        else:
            with self.connect() as c: c.execute("UPDATE states SET status='INACTIVE',valid_until=? WHERE state_id=?",(when,state_id))
    def clear_user(self,user_id):
        with self.connect() as c: c.execute("DELETE FROM transitions WHERE user_id=?",(user_id,)); c.execute("DELETE FROM states WHERE user_id=?",(user_id,))

class LazyDatabase:
    """Create the configured database only when it is first used."""

    def __init__(self):
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            self._instance = Database()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)


db = LazyDatabase()
