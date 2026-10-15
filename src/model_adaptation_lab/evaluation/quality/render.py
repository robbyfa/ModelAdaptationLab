"""Self-contained offline reviewer, without strategy mappings or diagnostic scores."""

import json
from pathlib import Path
from typing import Any

from model_adaptation_lab.evaluation.quality.packet import review_template, verify_packet
from model_adaptation_lab.io import LabError, atomic_write, digest


def render_packet(packet: Path, output: Path) -> None:
    manifest, cases = verify_packet(packet)
    if output.exists():
        raise LabError("Review HTML already exists")
    # Intern full source bodies once. Per-case availability and retrieved offsets stay intact.
    documents: dict[str, Any] = {}
    compact = []
    for case in cases:
        for doc in case["reference_documents"]:
            key = doc["document_id"]
            if key in documents and documents[key] != doc:
                raise LabError("Inconsistent reference document in packet")
            documents[key] = doc
        compact.append(
            case
            | {"reference_documents": [doc["document_id"] for doc in case["reference_documents"]]}
        )
    template = review_template(packet, manifest, cases)
    template["independent"] = False
    payload = json.dumps(
        {
            "cases": compact,
            "documents": documents,
            "template": template,
            "workspace": digest(str(output.resolve()).encode()),
            "rubric": (packet / "rubric.md").read_text(),
        },
        ensure_ascii=True,
    )
    # JSON script data must never terminate its element, even for adversarial model text.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    root = Path(__file__).parent
    html = (
        (root / "reviewer.html")
        .read_text()
        .replace("__SCRIPT__", (root / "reviewer.js").read_text())
        .replace("__PAYLOAD__", payload)
    )
    atomic_write(output, html.encode())
