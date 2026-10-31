"""Turn source documents into citable passages ("chunks").

Every chunk keeps where it came from (document, page, URL) so a retrieved
passage can always be cited and checked against the original.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCES_FILE = "sources.json"  # written by fetch_corpus.py


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str  # URL or citation
    page: int | None
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_space(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # re-join words hyphenated at line breaks
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    """Split text into overlapping windows of about ``size`` words."""
    if overlap >= size:
        raise ValueError("`overlap` must be smaller than `size`.")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def chunk_pages(doc_id: str, title: str, source: str, pages: list[str],
                size: int = 180, overlap: int = 40) -> list[Chunk]:
    """Chunk a document page by page, so every chunk has one page number."""
    chunks = []
    for p, raw in enumerate(pages, start=1):
        for i, piece in enumerate(chunk_words(normalize_space(raw), size, overlap)):
            chunks.append(Chunk(f"{doc_id}:p{p}:{i}", doc_id, title, source, p, piece))
    return chunks


def read_pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader  # imported lazily: only needed when building the index

    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def load_corpus(corpus_dir: str | Path, size: int = 180, overlap: int = 40) -> tuple[list[Chunk], dict]:
    """Load every source listed in ``sources.json`` and return chunks plus a manifest
    (file hashes) that identifies exactly which corpus an index was built from."""
    corpus_dir = Path(corpus_dir)
    meta_path = corpus_dir / SOURCES_FILE
    if not meta_path.is_file():
        raise ValueError(f"{meta_path} not found. Run `python rag_eval/fetch_corpus.py` first.")
    sources = json.loads(meta_path.read_text())
    chunks: list[Chunk] = []
    manifest = {"chunk_size": size, "chunk_overlap": overlap, "files": {}}
    for s in sources:
        path = corpus_dir / s["file"]
        manifest["files"][s["file"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        if s["kind"] == "pdf":
            chunks += chunk_pages(s["doc_id"], s["title"], s["url"], read_pdf_pages(path), size, overlap)
        elif s["kind"] == "pubmed":
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                text = normalize_space(f"{rec['title']} {rec['abstract']}")
                cite = f"PMID {rec['pmid']}; {rec.get('journal', '')} {rec.get('year', '')}".strip()
                chunks.append(Chunk(f"pmid:{rec['pmid']}", f"pmid:{rec['pmid']}", rec["title"],
                                    cite, None, text))
        else:
            raise ValueError(f"Unknown source kind {s['kind']!r} for {s['file']}.")
    manifest["n_chunks"] = len(chunks)
    manifest["corpus_sha256"] = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    return chunks, manifest
