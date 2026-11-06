"""Check the end-to-end study against the real ``memory-layer`` binary.

``endtoend.py`` re-implements the proxy's retrieval and formatting in Python.
This module removes that assumption: it loads every question's haystack into a
proxy SQLite database (one agent namespace per question, cached embeddings,
timestamps that reproduce each memory's age), starts the compiled proxy in
front of a recording shim, sends the questions through the proxy, and compares

* the prompt the proxy forwards with the prompt ``endtoend.py`` would send, and
* the answers with the simulated run.

The shim forwards every request to llama-server, so the answers are real.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from conformal_retrieval.endtoend import (
    CHAT_URL,
    MAX_TOKENS,
    SETTINGS,
    SYSTEM_PROMPT,
    build_messages,
    is_correct,
    results_path,
)
from conformal_retrieval.followup import short_answer
from conformal_retrieval.study import (
    MAX_EMBED_CHARS,
    QUESTIONS_PATH,
    SEED,
    EmbeddedQuestion,
    embedded_questions,
    pair_memories,
    parse_date,
)

LOGGER = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SHIM_PORT = 8492
PROXY_PORT = 8801
SCHEMA = """
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT, created_at REAL NOT NULL,
    updated_at REAL NOT NULL, user_text TEXT NOT NULL, assist_text TEXT NOT NULL,
    user_emb BLOB NOT NULL, assist_emb BLOB NOT NULL, access_count INTEGER DEFAULT 1
);
CREATE INDEX idx_memories_agent ON memories(agent_id);
CREATE INDEX idx_memories_created ON memories(created_at DESC);
"""


@dataclass(frozen=True)
class ProxyRun:
    """A study setting and the proxy flags that should reproduce it."""

    setting: str
    flags: tuple[str, ...]


RUNS = (
    ProxyRun("k5_b400", ("--top-k", "5")),
    ProxyRun("adaptive90_floor8_b400", ("--target-coverage", "0.9", "--adaptive-k")),
)


def unit(vector: np.ndarray) -> bytes:
    """Return the L2-normalized float32 bytes the proxy stores."""
    vector = np.asarray(vector, dtype=np.float32)
    return bytes((vector / np.linalg.norm(vector)).astype(np.float32).tobytes())


def build_database(
    path: Path, questions: Sequence[EmbeddedQuestion], now: float
) -> int:
    """Write every haystack as one agent namespace; return the memory count."""
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    rows = []
    for item in questions:
        question = item.question
        for index, memory in enumerate(question.memories):
            created = now - memory.age_hours * 3600.0
            rows.append(
                (
                    question.question_id,
                    created,
                    created,
                    memory.user_text,
                    memory.assistant_text,
                    unit(item.user_embeddings[index]),
                    unit(item.assistant_embeddings[index]),
                )
            )
    connection.executemany(
        "INSERT INTO memories (agent_id, created_at, updated_at, user_text,"
        " assist_text, user_emb, assist_emb) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()
    return len(rows)


def request_key(messages: Sequence[Mapping[str, Any]]) -> str:
    """Identify a request by its system date line and user question."""
    system = str(messages[0]["content"])
    date_line = next(line for line in system.splitlines() if "Current date" in line)
    return date_line + "\n" + str(messages[-1]["content"])


class Recorder:
    """Collect forwarded requests and backend replies by request key."""

    def __init__(self, backend: str) -> None:
        """Forward to ``backend`` (an OpenAI chat-completions URL)."""
        self.backend = backend
        self.lock = threading.Lock()
        self.records: dict[str, dict[str, Any]] = {}

    def handle(self, body: bytes) -> bytes:
        """Forward one request body and record it with the reply."""
        payload = json.loads(body)
        request = urllib.request.Request(
            self.backend,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            raw: bytes = response.read()
        reply = json.loads(raw)
        with self.lock:
            self.records[request_key(payload["messages"])] = {
                "messages": payload["messages"],
                "reply": str(reply["choices"][0]["message"]["content"] or ""),
            }
        return raw


def serve_recorder(recorder: Recorder, port: int) -> ThreadingHTTPServer:
    """Start the recording shim in a background thread."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                raw = recorder.handle(body)
                status = 200
            except Exception as error:  # noqa: BLE001 - report to the proxy
                LOGGER.warning("Shim forward failed: %s", error)
                raw, status = json.dumps({"error": str(error)}).encode(), 502
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silence per-request logging."""

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def wait_for(url: str, timeout: float = 120.0) -> None:
    """Poll ``url`` until it answers 200."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)
    raise RuntimeError(f"{url} did not become healthy")


def send(question: EmbeddedQuestion, date: str) -> None:
    """Send the bare question through the proxy, as a client would."""
    body = {
        "messages": [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nCurrent date: {date}."},
            {"role": "user", "content": question.question.text},
        ],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Agent-Id": question.question.question_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        response.read()


def memory_lines(system: str) -> list[str]:
    """Return injected lines with the relative age stripped."""
    return [
        line.split("] ", 1)[1]
        for line in system.splitlines()
        if line.startswith("- [") and "] " in line
    ]


def compare(
    expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]
) -> dict[str, bool | int]:
    """Compare forwarded and simulated prompts exactly and by memory lines."""
    want = memory_lines(str(expected[0]["content"]))
    got = memory_lines(str(actual[0]["content"]))
    return {
        "exact": list(expected) == list(actual),
        "same_lines": want == got,
        "same_set": sorted(want) == sorted(got),
        "expected_lines": len(want),
        "actual_lines": len(got),
    }


def run_setting(
    run: ProxyRun,
    questions: Sequence[EmbeddedQuestion],
    dates: Mapping[str, str],
    binary: Path,
    embedding_model: Path,
    workers: int,
    recorder: Recorder,
) -> None:
    """Start a fresh proxy for ``run`` and send every question through it."""
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "memories.sqlite"
        count = build_database(database, questions, time.time())
        LOGGER.info("%s: wrote %d memories", run.setting, count)
        command = [
            str(binary),
            "--embedding-model",
            str(embedding_model),
            "--backend",
            f"http://127.0.0.1:{SHIM_PORT}",
            "--port",
            str(PROXY_PORT),
            "--db",
            str(database),
            "--similarity-threshold",
            "-1",
            "--inject-mode",
            "system",
            *run.flags,
        ]
        log = (Path(directory) / "proxy.log").open("w")
        proxy = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for(f"http://127.0.0.1:{PROXY_PORT}/health")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = [
                    pool.submit(send, q, dates[q.question.question_id])
                    for q in questions
                ]
                for index, job in enumerate(jobs, start=1):
                    job.result()
                    if index % 50 == 0:
                        LOGGER.info("%s: %d/%d", run.setting, index, len(jobs))
        finally:
            proxy.terminate()
            proxy.wait(timeout=60)
            log.close()
            shutil.copy(Path(directory) / "proxy.log", f"/tmp/proxy-{run.setting}.log")


def main() -> None:
    """Run every proxy setting and write ``results/proxy_check_<model>.json``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="label used by endtoend.py")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--backend", default=CHAT_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()

    records = json.loads(QUESTIONS_PATH.read_text())
    answers = {str(r["question_id"]): str(r["answer"]) for r in records}
    dates = {
        str(r["question_id"]): parse_date(str(r["question_date"])).strftime("%Y-%m-%d")
        for r in records
    }
    questions = [
        question
        for question in embedded_questions([pair_memories(r) for r in records])
        if not question.question.abstention
        and short_answer(question, answers) is not None
    ]
    if arguments.limit:
        questions = questions[: arguments.limit]
    keys = {
        request_key(build_messages(q, SETTINGS[0], dates[q.question.question_id]))
        for q in questions
    }
    if len(keys) != len(questions):
        raise RuntimeError("Request keys are not unique across questions")

    simulated = json.loads(results_path(arguments.model).read_text())["settings"]
    settings = {setting.name: setting for setting in SETTINGS}
    summary: dict[str, Any] = {
        "model": arguments.model,
        "questions": len(questions),
        "max_embed_chars": MAX_EMBED_CHARS,
        "settings": {},
    }
    for run in RUNS:
        recorder = Recorder(arguments.backend)
        server = serve_recorder(recorder, SHIM_PORT)
        try:
            run_setting(
                run,
                questions,
                dates,
                arguments.binary,
                arguments.embedding_model,
                arguments.workers,
                recorder,
            )
        finally:
            server.shutdown()
            server.server_close()
        rows = []
        for question in questions:
            qid = question.question.question_id
            expected = build_messages(question, settings[run.setting], dates[qid])
            record = recorder.records.get(request_key(expected))
            if record is None:
                raise RuntimeError(f"No forwarded request recorded for {qid}")
            row: dict[str, Any] = dict(compare(expected, record["messages"]))
            row["question_id"] = qid
            row["correct"] = is_correct(answers[qid], record["reply"])
            rows.append(row)
        reference = simulated[run.setting]
        summary["settings"][run.setting] = {
            "flags": list(run.flags),
            "exact_prompt": float(np.mean([r["exact"] for r in rows])),
            "same_lines_in_order": float(np.mean([r["same_lines"] for r in rows])),
            "same_line_set": float(np.mean([r["same_set"] for r in rows])),
            "mean_lines_proxy": float(np.mean([r["actual_lines"] for r in rows])),
            "mean_lines_simulated": float(np.mean([r["expected_lines"] for r in rows])),
            "accuracy_proxy": float(np.mean([r["correct"] for r in rows])),
            "accuracy_simulated": float(reference["accuracy"])
            if len(questions) == simulated_questions(arguments.model)
            else None,
            "mismatched_questions": [r["question_id"] for r in rows if not r["exact"]],
        }
        LOGGER.info("%s: %s", run.setting, summary["settings"][run.setting])
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"proxy_check_{results_path(arguments.model).stem[10:]}.json"
    output.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Wrote %s", output)


def simulated_questions(model: str) -> int:
    """Return how many questions the simulated run for ``model`` used."""
    return int(json.loads(results_path(model).read_text())["questions"])


if __name__ == "__main__":
    main()
