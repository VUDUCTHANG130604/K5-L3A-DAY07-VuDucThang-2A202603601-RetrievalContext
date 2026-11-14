"""Serve the local Retrieval Lab dashboard and read-only project data API."""
from __future__ import annotations

import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
PORT = 8765


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def build_payload():
    data_dir = ROOT / "data"
    result_dir = ROOT / "starter" / "results"
    cases = read_json(data_dir / "cases.json", [])
    questions = read_json(data_dir / "questions.json", [])
    chunks = read_json(data_dir / "chunks.json", [])
    retrieval = {}
    agent = {}
    agent_traces = {}
    agent_reviews = {}

    for split in ("dev", "test"):
        eval_path = result_dir / f"eval_{split}_budget1200"
        retrieval[split] = {
            "summary": read_json(eval_path / "summary.json", {}),
            "traces": read_json(eval_path / "traces.json", {}),
        }
        agent_path = result_dir / f"agent_{split}"
        agent[split] = read_json(agent_path / "summary.json", None)
        agent_traces[split] = read_json(agent_path / "traces.json", [])
        agent_reviews[split] = read_json(agent_path / "reviews.json", [])

    payload = {
        "cases": cases,
        "questions": questions,
        "chunks": chunks,
        "retrieval": retrieval,
        "agent": agent,
        "agentTraces": agent_traces,
        "agentReviews": agent_reviews,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if urlsplit(self.path).path == "/api/dashboard":
            body = build_payload()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        requested = unquote(urlsplit(self.path).path)
        if requested == "/" or requested == "/dashboard":
            self.path = "/dashboard/index.html"
        super().do_GET()

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


if __name__ == "__main__":
    print(f"Retrieval Lab dashboard: http://127.0.0.1:{PORT}")
    print("Press Ctrl+C to stop.")
    ThreadingHTTPServer(("127.0.0.1", PORT), DashboardHandler).serve_forever()
