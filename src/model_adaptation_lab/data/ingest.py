"""Strict adapters for the verified IBM and NVIDIA schemas.

NVIDIA annotations are inspection-only. They are never a training or corpus fallback.
"""

import re
import tarfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from model_adaptation_lab.io import LabError, atomic_write, file_hash, read_json


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    answer: str | None
    answerable: bool
    document_id: str | None
    candidate_document_ids: tuple[str, ...]
    source: str
    split: str
    start: int | None = None
    end: int | None = None

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    source_url: str | None
    source: str = "original_technotes"


def string(row: dict[str, Any], key: str, *, empty: bool = False) -> str:
    value = row.get(key)
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise LabError(f"{key}: expected {'possibly empty ' if empty else ''}string")
    return value


def objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(r, dict) for r in value):
        raise LabError("Expected a JSON array of objects")
    return value


def original_questions(path: Path, split: str) -> list[Question]:
    if split not in {"train", "validation", "test"}:
        raise LabError(f"Unknown split: {split}")
    result = []
    for row in objects(read_json(path)):
        qid = string(row, "QUESTION_ID")
        title = string(row, "QUESTION_TITLE", empty=True)
        # IBM's README says BODY in one section; actual baseline consumes TEXT.
        body_key = "QUESTION_TEXT" if "QUESTION_TEXT" in row else "QUESTION_BODY"
        body = string(row, body_key, empty=True)
        question = (title + "\n\n" + body).strip()
        if not question:
            raise LabError(f"{qid}: empty question")
        flag = row.get("ANSWERABLE")
        if flag not in ("Y", "N"):
            raise LabError(f"{qid}: ANSWERABLE must be Y or N; unlabeled test is unsupported")
        ids = row.get("DOC_IDS")
        if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids):
            raise LabError(f"{qid}: DOC_IDS must be a list of nonempty strings")
        answer: str | None = string(row, "ANSWER", empty=True)
        doc: str | None = string(row, "DOCUMENT")
        start, end = row.get("START_OFFSET"), row.get("END_OFFSET")
        if flag == "Y":
            if answer == "-" or doc == "-" or doc not in ids:
                raise LabError(f"{qid}: answerable row lacks answer or candidate gold document")
            if not all(isinstance(v, str) and re.fullmatch(r"[0-9]+", v) for v in (start, end)):
                raise LabError(f"{qid}: expected decimal-string answer offsets")
            assert isinstance(start, str) and isinstance(end, str)
            start, end = int(start), int(end)
            if not 0 <= start < end:
                raise LabError(f"{qid}: invalid answer offsets")
        else:
            if any(v != "-" for v in (answer, doc, start, end)):
                raise LabError(f"{qid}: unanswerable row must use '-' annotation sentinels")
            answer, doc, start, end = None, None, None, None
        result.append(
            Question(
                qid, question, answer, flag == "Y", doc, tuple(ids), "original", split, start, end
            )
        )
    unique_ids(result)
    return result


def unique_ids(rows: list[Question]) -> None:
    ids = [r.question_id for r in rows]
    if len(ids) != len(set(ids)):
        raise LabError("Duplicate question IDs within a source; assignments would be ambiguous")


def original_documents(path: Path) -> dict[str, Document]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise LabError("Original corpus must be an object keyed by document ID")
    documents = {}
    for key, row in raw.items():
        if not isinstance(row, dict) or string(row, "id") != key:
            raise LabError(f"Document key/id mismatch: {key}")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise LabError(f"{key}: metadata must be an object")
        url = metadata.get("canonicalUrl")
        if url is not None and not isinstance(url, str):
            raise LabError(f"{key}: canonicalUrl must be a string")
        documents[key] = Document(
            key, string(row, "title", empty=True), string(row, "text", empty=True), url
        )
    return documents


def nvidia_questions(path: Path) -> tuple[list[Question], list[Document]]:
    questions, contexts = [], []
    for row in objects(read_json(path)):
        qid, question = (string(row, k) for k in ("id", "question"))
        answer = string(row, "answer", empty=True)
        flag = row.get("is_impossible")
        if type(flag) is not bool:
            raise LabError(f"{qid}: is_impossible must be boolean")
        if (flag and answer != "-") or (not flag and answer == "-"):
            raise LabError(f"{qid}: answer contradicts is_impossible")
        docs = []
        for ctx in objects(row.get("contexts")):
            filename, text = string(ctx, "filename"), string(ctx, "text")
            docs.append(filename.removesuffix(".txt"))
            contexts.append(Document(docs[-1], "", text, None, "nvidia_annotation_context"))
        questions.append(
            Question(
                qid,
                question,
                None if flag else answer,
                not flag,
                None,
                tuple(docs),
                "nvidia",
                "inspection_only",
            )
        )
    unique_ids(questions)
    return questions, contexts


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise LabError(f"Cannot load configuration {path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise LabError("Expected dataset config schema_version: 1")
    for section in ("original", "nvidia", "ibm_code", "policy"):
        if not isinstance(config.get(section), dict):
            raise LabError(f"{section}: expected configuration object")
    for source in ("original", "nvidia", "ibm_code"):
        revision = config.get(source, {}).get("revision", "")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise LabError(f"{source}: immutable 40-character revision required")
    for source in ("original", "nvidia"):
        entry = config[source]
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry["sha256"]
        ):
            raise LabError(f"{source}: expected SHA-256 checksum")
        for key in ("repository", "file"):
            string(entry, key)
        if Path(entry["file"]).name != entry["file"]:
            raise LabError("Source filename must not contain a path")
    if config["nvidia"].get("role") != "inspection_only":
        raise LabError("NVIDIA derivative must remain inspection_only")
    threshold = config.get("policy", {}).get("near_duplicate_threshold")
    if type(threshold) not in (int, float) or not 0 < threshold <= 1:
        raise LabError("near_duplicate_threshold must be in (0, 1]")
    if config["policy"].get("split_source") != "official":
        raise LabError("Only official split assignments are supported in Phase 1")
    return config


def download(url: str, path: Path, expected_hash: str | None = None) -> None:
    if path.exists():
        if expected_hash and file_hash(path) != expected_hash:
            raise LabError(f"Checksum mismatch: {path}; remove the corrupt file and retry")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if expected_hash and file_hash(part) != expected_hash:
            raise LabError(f"Downloaded checksum mismatch: {url}")
        part.replace(path)
    except (OSError, urllib.error.URLError) as exc:
        raise LabError(
            f"Download failed: {url}: {exc}. Supply local source files instead."
        ) from exc
    finally:
        part.unlink(missing_ok=True)


def extract_original(archive: Path, destination: Path) -> list[str]:
    """Read only allowlisted regular files; never extract paths, links, or the huge corpus."""
    wanted = {
        "training_Q_A.json",
        "dev_Q_A.json",
        "training_dev_technotes.json",
        "README.txt",
        "CDLA-Permissive-v1.0.pdf",
    }
    found: list[str] = []
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = Path(member.name).name
            # README/license copies can exist in multiple folders; only top-level terms.
            if name not in wanted or not member.isfile():
                continue
            if name in found:
                if name in {"README.txt", "CDLA-Permissive-v1.0.pdf"}:
                    continue
                raise LabError(f"Ambiguous archive member: {name}")
            if member.size > 512 * 1024 * 1024:
                raise LabError(f"Unexpectedly large archive member: {member.name}")
            source = stream.extractfile(member)
            if source is not None:
                atomic_write(destination / name, source.read())
                found.append(name)
    if not {"training_Q_A.json", "dev_Q_A.json", "training_dev_technotes.json"} <= set(found):
        raise LabError("Official archive lacks expected training/development files")
    return found


def fetch_sources(config: dict[str, Any], raw: Path) -> None:
    for name in ("original", "nvidia"):
        src = config[name]
        url = (
            f"https://huggingface.co/datasets/{src['repository']}/resolve/"
            f"{src['revision']}/{src['file']}"
        )
        path = raw / name / src["file"]
        download(url, path, src.get("sha256"))
        if name == "original":
            extract_original(path, raw / name)
