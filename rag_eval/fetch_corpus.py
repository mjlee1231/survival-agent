"""Download the retrieval corpus into data/corpus/.

- Three FDA guidance documents (U.S. government works, public domain).
- PubMed titles and abstracts on survival-analysis methods in trials, via the
  NCBI E-utilities API (no key needed; please follow NCBI's usage policy).

The corpus is not committed to git: abstracts are copyrighted by their
publishers, and anyone can rebuild the same corpus with this script.

    python rag_eval/fetch_corpus.py --email you@example.com
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"

GUIDANCE = [
    {
        "doc_id": "fda_e9",
        "title": "E9 Statistical Principles for Clinical Trials (FDA/ICH, 1998)",
        "url": "https://www.fda.gov/media/71336/download",
        "file": "fda_e9.pdf",
    },
    {
        "doc_id": "fda_e9r1",
        "title": "E9(R1) Statistical Principles for Clinical Trials: Addendum on Estimands and "
                 "Sensitivity Analysis (FDA/ICH, 2021)",
        "url": "https://www.fda.gov/media/148473/download",
        "file": "fda_e9r1.pdf",
    },
    {
        "doc_id": "fda_covariates",
        "title": "Adjusting for Covariates in Randomized Clinical Trials for Drugs and Biological "
                 "Products (FDA, 2023)",
        "url": "https://www.fda.gov/media/148910/download",
        "file": "fda_covariates.pdf",
    },
]

PUBMED_QUERIES = [
    '"proportional hazards" assumption randomized trial',
    '"restricted mean survival time" clinical trial',
    'estimand "intercurrent events"',
    'covariate adjustment randomized trial "hazard ratio"',
    'sample size survival trial "log-rank"',
    '"non-proportional hazards" "delayed treatment effect"',
]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
HEADERS = {"User-Agent": "Mozilla/5.0 (survagent corpus builder; research use)"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_guidance() -> list[dict]:
    sources = []
    for g in GUIDANCE:
        path = CORPUS / g["file"]
        if path.exists():
            print(f"  already have {g['file']}")
        else:
            print(f"  downloading {g['title'][:60]}...")
            try:
                data = get(g["url"])
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {exc}\n  Download {g['url']} in a browser and save it as {path}")
                continue
            if not data.startswith(b"%PDF"):
                print(f"  FAILED: {g['url']} did not return a PDF.\n"
                      f"  Download it in a browser and save it as {path}")
                continue
            path.write_bytes(data)
        sources.append({"kind": "pdf", **g})
    return sources


def fetch_pubmed(email: str | None, per_query: int) -> dict:
    extra = {"tool": "survagent", **({"email": email} if email else {})}
    pmids: list[str] = []
    for q in PUBMED_QUERIES:
        params = {"db": "pubmed", "term": q + " AND hasabstract[text] AND english[lang]",
                  "retmax": per_query, "sort": "relevance", "retmode": "json", **extra}
        res = json.loads(get(EUTILS + "esearch.fcgi?" + urllib.parse.urlencode(params)))
        ids = res["esearchresult"]["idlist"]
        print(f"  {len(ids):3d} abstracts for {q}")
        pmids += [i for i in ids if i not in pmids]
        time.sleep(0.4)  # stay under NCBI's 3 requests/second

    records = []
    for start in range(0, len(pmids), 100):
        batch = pmids[start:start + 100]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", **extra}
        root = ET.fromstring(get(EUTILS + "efetch.fcgi?" + urllib.parse.urlencode(params)))
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = "".join(art.find(".//ArticleTitle").itertext()).strip()
            parts = []
            for a in art.findall(".//Abstract/AbstractText"):
                label = a.get("Label")
                text = "".join(a.itertext()).strip()
                parts.append(f"{label}: {text}" if label else text)
            year = art.findtext(".//JournalIssue/PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
            records.append({"pmid": pmid, "title": title, "abstract": " ".join(parts),
                            "journal": art.findtext(".//Journal/ISOAbbreviation") or "", "year": year[:4]})
        time.sleep(0.4)

    path = CORPUS / "pubmed.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  saved {len(records)} abstracts")
    return {"kind": "pubmed", "doc_id": "pubmed", "title": "PubMed abstracts", "file": path.name,
            "url": EUTILS, "queries": PUBMED_QUERIES, "retrieved": date.today().isoformat()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", help="your email, sent to NCBI as their usage policy asks")
    ap.add_argument("--per-query", type=int, default=40)
    ap.add_argument("--skip-pubmed", action="store_true")
    args = ap.parse_args()
    CORPUS.mkdir(parents=True, exist_ok=True)
    print("FDA guidance:")
    sources = fetch_guidance()
    if not args.skip_pubmed:
        print("PubMed:")
        sources.append(fetch_pubmed(args.email, args.per_query))
    (CORPUS / "sources.json").write_text(json.dumps(sources, indent=2))
    print(f"Wrote {CORPUS / 'sources.json'} ({len(sources)} sources)")
    if len(sources) < len(GUIDANCE) + (0 if args.skip_pubmed else 1):
        sys.exit("Some sources are missing; see messages above, then run this script again.")


if __name__ == "__main__":
    main()
