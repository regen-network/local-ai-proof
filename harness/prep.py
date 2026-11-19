"""Build an evidence-location task from a past review: a corpus of passages plus gold labels.

The gold labels come from the HUMAN registry reviewer's signed checklist (.docx): for every
requirement, the "Submitted Material" column names the document section(s) the reviewer
accepted as evidence. The task: given the requirement text, find those sections.

Two corpus shapes:
  --kind sections  : a Markdown project document split on numbered headings (1.1, 1.6.1, 2, ...)
  --kind pages     : a PDF report split by page (pdftotext); gold = pages carrying the named
                     sections / tables / figures the reviewer cited (headings and captions only).

Outputs go to private/ (gitignored): real review content never enters the repo.

  python3 harness/prep.py --kind sections --checklist REVIEW.docx --doc PLAN.md --out private/tasks/reg.json
  python3 harness/prep.py --kind pages    --checklist REVIEW.docx --doc REPORT.pdf --out private/tasks/iss.json
"""
from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from xml.etree import ElementTree as ET

from common import write_json

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def checklist_rows(path: str) -> list[dict]:
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    table = root.findall(f".//{W}tbl")[-1]
    rows, cat = [], ""
    for tr in table.findall(f"{W}tr"):
        cells = []
        for tc in tr.findall(f"{W}tc"):
            paras = ["".join(x.text or "" for x in p.iter(W + "t")).strip() for p in tc.iter(W + "p")]
            cells.append(" / ".join(p for p in paras if p))
        if len(cells) < 6 or cells[1] == "Requirement":
            continue
        cat = cells[0] or cat
        rows.append({"category": cat, "requirement": cells[1], "accepted_evidence": cells[2],
                     "submitted_material": cells[3], "verdict": cells[4], "comments": cells[5]})
    return rows


def primary_part(s: str) -> str:
    s = s.split("Supplementary Evidence")[0].split("Secondary Evidence")[0]
    return s.replace("Primary Documentation:", "")


# ------------------------------------------------------------------ sections (Markdown)

def md_sections(path: str) -> list[dict]:
    secs, cur = [], None
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.match(r"^#{1,4}\s*\**\s*((?:\d+\.)+\d*|Appendix)\.?\s+(.*)", line.strip())
        m2 = re.match(r"^#{1,4}\s*\**\s*(Appendix)\b(.*)", line.strip())
        m = m or m2
        if m:
            sid = m.group(1).rstrip(".")
            cur = {"sid": sid, "title": m.group(2).strip(" *#"), "text": ""}
            secs.append(cur)
        elif cur is not None:
            cur["text"] += line
    for s in secs:
        s["text"] = (s["title"] + "\n" + s["text"]).strip()
    return [s for s in secs if len(s["text"]) > 40]


def gold_sections(material: str) -> list[str]:
    p = primary_part(material)
    if not re.search(r"Section|Appendix", p):
        return []
    ids = re.findall(r"\b(\d+(?:\.\d+)*)\b", p)
    out = [i for i in ids if re.match(r"^[1-9](\.\d{1,2})*$", i)]
    if "Appendix" in p:
        out.append("Appendix")
    return sorted(set(out))


# ------------------------------------------------------------------ pages (PDF)

NAMED = ["Executive Summary", "Project Outline", "Credit Statement", "Carbon Balance", "Sampling Method",
         "Laboratory Analysis", "AI Processing", "Accuracy", "Greenhouse Gas Emissions Monitoring",
         "Leakage Deduction", "Monitored Impact Information", "SOCS Calculation"]


def pdf_pages(path: str) -> list[dict]:
    txt = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True).stdout
    pages = []
    for i, p in enumerate(txt.split("\f")):
        if len(p.strip()) < 40:
            continue
        toc = len(re.findall(r"\.{6,}", p)) >= 4
        pages.append({"sid": f"p{i + 1}", "title": f"page {i + 1}", "text": p.strip(), "toc": toc})
    return pages


def gold_pages(material: str, pages: list[dict]) -> list[str]:
    p = primary_part(material)
    body = [pg for pg in pages if not pg["toc"]]
    gold = set()
    for name in NAMED:
        cited = re.search(re.escape(name), p, re.I) or (name == "Leakage Deduction" and re.search("Leakage", p))
        if cited:
            for pg in body:
                # heading = a short line that is (nearly) just the name
                if any(re.fullmatch(rf"\s*{re.escape(name)}\s*:?\s*", ln, re.I) for ln in pg["text"].splitlines()):
                    gold.add(pg["sid"])
    for kind, num in re.findall(r"\b(Table|Figure)\s*(\d+)", p):
        for pg in body:
            if re.search(rf"^\s*{kind}\s*{num}\s*[:\-–]", pg["text"], re.M):
                gold.add(pg["sid"])
    return sorted(gold, key=lambda s: int(s[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sections", "pages"], required=True)
    ap.add_argument("--checklist", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = checklist_rows(a.checklist)
    corpus = md_sections(a.doc) if a.kind == "sections" else pdf_pages(a.doc)
    items = []
    for i, r in enumerate(rows):
        g = gold_sections(r["submitted_material"]) if a.kind == "sections" else gold_pages(r["submitted_material"], corpus)
        items.append({"id": f"R{i + 1:02d}", **r, "gold": g})
    if a.kind == "pages":
        corpus = [c for c in corpus if not c["toc"]]
    write_json(a.out, {"kind": a.kind, "corpus": corpus, "items": items})
    n = sum(1 for x in items if x["gold"])
    print(f"{len(items)} checklist rows, {n} with gold locations, corpus {len(corpus)} passages, "
          f"verdicts: {sorted(set(x['verdict'] for x in items))}")


if __name__ == "__main__":
    main()
