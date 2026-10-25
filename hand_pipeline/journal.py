"""Transactional episode boundaries; interrupted rollouts never become task failures."""
import json
from pathlib import Path
import sqlite3
import time


class Journal:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, signature TEXT NOT NULL)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS episodes
            (job TEXT, episode INTEGER, status TEXT, payload TEXT, updated REAL,
             PRIMARY KEY(job, episode), FOREIGN KEY(job) REFERENCES jobs(id))''')
        self.db.commit()

    def register(self, job, signature):
        row = self.db.execute('SELECT signature FROM jobs WHERE id=?', (job,)).fetchone()
        if row and row[0] != signature:
            raise ValueError(f'Protocol/checkpoint changed for {job}; use a new result namespace')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO jobs VALUES (?,?)', (job, signature))

    def commit(self, job, episode, status, payload):
        if status not in ('SUCCESS', 'TASK_FAILURE', 'RUNTIME_FAILURE', 'ACTION_EXECUTION_INVALID'):
            raise ValueError(status)
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?)',
                (job, int(episode), status, json.dumps(payload), time.time()))

    def completed(self, job):
        return {row[0] for row in self.db.execute(
            "SELECT episode FROM episodes WHERE job=? AND status IN ('SUCCESS','TASK_FAILURE')", (job,))}

    def summary(self, job):
        counts = dict(self.db.execute('SELECT status,COUNT(*) FROM episodes WHERE job=? GROUP BY status', (job,)))
        total = counts.get('SUCCESS', 0) + counts.get('TASK_FAILURE', 0)
        return dict(completed=total, successes=counts.get('SUCCESS', 0),
            success_rate=counts.get('SUCCESS', 0) / total if total else None, counts=counts)

    def close(self):
        self.db.close()
