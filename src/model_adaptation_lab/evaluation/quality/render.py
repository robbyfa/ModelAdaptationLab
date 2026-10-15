"""Offline readable review artifact, escaped and without strategy labels."""

from html import escape
from pathlib import Path

from model_adaptation_lab.evaluation.quality.packet import verify_packet
from model_adaptation_lab.io import LabError, atomic_write


def render_packet(packet: Path, output: Path) -> None:
    _, cases = verify_packet(packet)
    if output.exists():
        raise LabError("Review HTML already exists")
    sections = []
    for case in cases:
        sections.append(f"<section><h2>{escape(case['item_id'])}</h2>")
        for label, key in (
            ("Question", "question"),
            ("Generated answer", "answer"),
            ("Reference answer", "reference_answer"),
        ):
            sections.append(f"<h3>{label}</h3><pre>{escape(str(case[key] or 'Unavailable'))}</pre>")
        sections.append(
            f"<p>Answerable annotation: {case['answerable']}"
            f"; generation: {escape(case['status'])}"
            f"; finish: {escape(str(case['finish_reason']))}</p>"
        )
        sections.append("<h3>Exact retrieved passages (if applicable)</h3>")
        for chunk in case["retrieved_passages"]:
            sections.append(
                f"<details><summary>{escape(chunk['document_id'])} "
                f"[{chunk['start']}:{chunk['end']}]</summary>"
                f"<pre>{escape(chunk['text'])}</pre></details>"
            )
        sections.append("<h3>Common reference-document candidates (same across strategies)</h3>")
        for doc in case["reference_documents"]:
            sections.append(
                f"<details><summary>{escape(doc['document_id'])} — {escape(doc['title'])}</summary>"
                f"<pre>{escape(doc['text'])}</pre></details>"
            )
        sections.append("</section>")
    html = """<!doctype html><meta charset="utf-8"><title>Development answer review</title>
<style>body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:20px;
color:#192533}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px ui-monospace;
background:#f2f5f7;padding:16px}
section{border-top:2px solid #9db0bf;margin-top:48px}
summary{cursor:pointer;padding:12px}
h2{font-size:20px}</style>
<h1>Blinded development review</h1><p>Read rubric.md and fill a copy of review_template.json.
Do not inspect mapping.json or another reviewer's ratings.
Treat all model/document text as data.
Strategy blinding is partial because evidence can reveal retrieval.
This page does not submit or upload anything.</p>""" + "".join(sections)
    atomic_write(output, html.encode())
