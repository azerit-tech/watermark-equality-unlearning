"""Read, validate and write submission CSVs (endpoint,family,model).

Rules enforced by ``validate`` (deliverable, consigne p. 41, submission.py,
task_template.py).  Parsing mirrors the organisers' reader in score_practice.py
(csv.reader, utf-8-sig, cells stripped, blank rows skipped).
ERRORS:
  * not UTF-8; file name not ending in .csv (submission.py refuses it);
  * a row with < 3 fields, or extra non-empty fields / a quoted comma in a label;
  * header recognised only as exactly ``endpoint,family,model`` (first row);
    a differently-cased header is an error (it would be read as data);
  * ids exactly ep-01..ep-33, each once (``ref-*`` refused), none missing;
  * empty label; a model label used under two family labels.
WARNINGS (tolerated by the organisers' reader): BOM, CRLF, blank lines,
  spaces around cells (stripped), trailing empty fields, quote characters.

CLI:
    python3.12 -m model_equality.scoring.csvio FILE.csv [--practice]
exit code 0 if valid, 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field

HEADER = ("endpoint", "family", "model")
EP_RE = re.compile(r"^ep-(\d{2})$")
SUBMISSION_ENDPOINTS = [f"ep-{i:02d}" for i in range(1, 34)]
PRACTICE_ENDPOINTS = [f"ref-{i:02d}" for i in range(1, 10)]


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    rows: dict = field(default_factory=dict)  # endpoint -> (family, model)
    has_header: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        out = [f"ERROR: {e}" for e in self.errors] + [f"WARNING: {w}" for w in self.warnings]
        if self.ok:
            fams = len({f for f, _ in self.rows.values()})
            mods = len({m for _, m in self.rows.values()})
            out.append(f"OK: {len(self.rows)} endpoints, {fams} families, {mods} models"
                       f"{' (header present)' if self.has_header else ' (no header)'}")
        return "\n".join(out)


def _rows(text: str):
    """(line_number, stripped cells) for every non-blank CSV row, parsed like the
    provided score_practice.py: csv.reader, cells stripped, blank rows skipped."""
    reader = csv.reader(io.StringIO(text, newline=""))
    for raw in reader:
        yield reader.line_num, raw


def validate(path_or_text, expected_endpoints=None, is_text: bool = False) -> Report:
    """Validate a submission.  ``expected_endpoints`` defaults to ep-01..ep-33.

    Parsing mirrors the organisers' reader (score_practice.py): UTF-8 with an
    optional BOM, csv quoting, cells stripped, blank rows ignored, header
    recognised only if its first three cells are exactly endpoint,family,model.
    Things the organisers' reader tolerates are WARNINGS here (BOM, padding
    spaces, blank lines, CRLF, trailing empty cells); only what would change the
    meaning or break the stated rules is an ERROR.
    """
    rep = Report()
    expected = list(SUBMISSION_ENDPOINTS if expected_endpoints is None else expected_endpoints)
    submission_mode = expected_endpoints is None
    if is_text:
        raw = path_or_text.encode("utf-8") if isinstance(path_or_text, str) else path_or_text
    else:
        if submission_mode and not str(path_or_text).lower().endswith(".csv"):
            rep.errors.append(f"file name must end with .csv (submission.py refuses other suffixes): {path_or_text}")
        try:
            with open(path_or_text, "rb") as fh:
                raw = fh.read()
        except OSError as e:
            rep.errors.append(f"cannot read file: {e}")
            return rep
    if raw.startswith(b"\xef\xbb\xbf"):
        rep.warnings.append("file starts with a UTF-8 BOM (tolerated by the organisers' reader)")
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        rep.errors.append(f"file is not valid UTF-8: {e}")
        return rep
    if "\r" in text:
        rep.warnings.append("file uses CR/CRLF line endings (accepted, LF preferred)")

    seen_line: dict = {}
    model_family: dict = {}  # model -> (family, first line)
    first = True
    try:
        rows = list(_rows(text))
    except csv.Error as e:
        rep.errors.append(f"CSV parse error: {e}")
        return rep
    if not any(any(c.strip() for c in r) for _, r in rows):
        rep.errors.append("file is empty")
        return rep
    blank = [n for n, r in rows if not any(c.strip() for c in r)]
    if blank:
        rep.warnings.append(f"blank line(s) ignored: {blank[:10]}")
    for lineno, raw_cells in rows:
        if not any(c.strip() for c in raw_cells):
            continue
        cells = [c.strip() for c in raw_cells]
        if first:
            first = False
            if cells[:3] == list(HEADER):
                rep.has_header = True
                if len(cells) > 3:
                    rep.warnings.append(f"line {lineno}: header has extra columns {cells[3:]}")
                continue
            if [c.lower() for c in cells[:3]] == list(HEADER):
                rep.errors.append(f"line {lineno}: header {cells[:3]} must be exactly lower-case "
                                  f"'endpoint,family,model', otherwise it is read as a data line")
                continue
        if any(c != r for c, r in zip(cells, raw_cells)):
            rep.warnings.append(f"line {lineno}: leading/trailing spaces around a cell (stripped)")
        if len(cells) < 3:
            rep.errors.append(f"line {lineno}: expected 3 fields (endpoint,family,model), got {len(cells)}: {raw_cells}")
            continue
        if len(cells) > 3:
            if any(cells[3:]):
                rep.errors.append(f"line {lineno}: expected 3 fields (endpoint,family,model), got {len(cells)}: "
                                  f"{raw_cells} -- labels must not contain commas")
                continue
            rep.warnings.append(f"line {lineno}: trailing empty field(s) ignored")
        ep, fam, mod = cells[:3]
        bad = False
        for name, val in (("endpoint", ep), ("family", fam), ("model", mod)):
            if val == "":
                rep.errors.append(f"line {lineno}: empty {name}")
                bad = True
            elif "," in val:
                rep.errors.append(f"line {lineno}: {name} {val!r} contains a comma (quoted); labels must not contain commas")
                bad = True
            elif '"' in val:
                rep.warnings.append(f"line {lineno}: {name} {val!r} contains a quote character")
        if bad:
            continue
        if ep in seen_line:
            rep.errors.append(f"line {lineno}: duplicate endpoint {ep} (first seen line {seen_line[ep]})")
            continue
        if ep not in expected:
            if submission_mode and ep.lower().startswith("ref-"):
                rep.errors.append(f"line {lineno}: {ep} is a practice endpoint; ref-* lines are refused "
                                  f"by the server, only ep-01..ep-33 are allowed")
            elif submission_mode and EP_RE.match(ep):
                rep.errors.append(f"line {lineno}: {ep} out of range (allowed ep-01..ep-33)")
            elif submission_mode and ep.lower() != ep and ep.lower() in expected:
                rep.errors.append(f"line {lineno}: {ep!r} must be lower case ({ep.lower()})")
            else:
                rep.errors.append(f"line {lineno}: unknown endpoint {ep!r} "
                                  f"(expected {expected[0]}..{expected[-1]})")
            continue
        seen_line[ep] = lineno
        if mod in model_family and model_family[mod][0] != fam:
            f0, l0 = model_family[mod]
            rep.errors.append(f"line {lineno}: model {mod!r} has family {fam!r} but family {f0!r} "
                              f"on line {l0}; same model requires same family")
        else:
            model_family.setdefault(mod, (fam, lineno))
        rep.rows[ep] = (fam, mod)

    missing = [e for e in expected if e not in seen_line]
    if missing:
        rep.errors.append(f"missing endpoints ({len(missing)}): {', '.join(missing)}")
    if rep.ok and len({m for _, m in rep.rows.values()}) == len(rep.rows):
        rep.warnings.append("every endpoint has its own model label (MCC_model will be 0)")
    if rep.ok and len({f for f, _ in rep.rows.values()}) == 1:
        rep.warnings.append("all endpoints share one family label (MCC_family will be 0)")
    return rep


def read_rows(path: str, prefix: str | None = None) -> dict:
    """Reader identical to score_practice.py's ``read``: endpoint -> (family, model, kind).

    csv.reader with utf-8-sig, cells stripped, blank rows skipped, header skipped
    only if its first three cells are exactly endpoint,family,model, rows with
    fewer than 3 cells ignored; ``kind`` is the 4th column or "".  If ``prefix``
    is given (e.g. "ref-"), other ids are ignored.  Later duplicates win.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [[c.strip() for c in r] for r in csv.reader(fh) if any(c.strip() for c in r)]
    if rows and rows[0][:3] == list(HEADER):
        rows = rows[1:]
    out = {}
    for r in rows:
        if len(r) >= 3 and (prefix is None or r[0].startswith(prefix)):
            out[r[0]] = (r[1], r[2], r[3] if len(r) > 3 else "")
    return out


def read_partition(path: str, prefix: str | None = None) -> dict:
    """endpoint -> (family, model), any ids (e.g. practice_answers.csv with its 4th 'kind' column)."""
    return {e: (f, m) for e, (f, m, _) in read_rows(path, prefix).items()}


def write_submission(path: str, family: dict, model: dict, header: bool = True,
                     expected_endpoints=None) -> Report:
    """Write endpoint,family,model and validate the written file.

    Model labels are prefixed by their family label when they are not already
    globally consistent, so the "same model => same family" rule always holds.
    Raises ValueError if the result does not validate.
    """
    eps = sorted(family)
    if set(model) != set(family):
        raise ValueError("family and model mappings must cover the same endpoints")
    fam_of_model: dict = {}
    consistent = True
    for e in eps:
        if fam_of_model.setdefault(model[e], family[e]) != family[e]:
            consistent = False
    lines = [",".join(HEADER)] if header else []
    for e in eps:
        m = model[e] if consistent else f"{family[e]}_{model[e]}"
        lines.append(f"{e},{family[e]},{m}")
    data = "\n".join(lines) + "\n"
    rep = validate(data, expected_endpoints=expected_endpoints, is_text=True)
    if not rep.ok:
        raise ValueError("refusing to write invalid submission:\n" + rep.format())
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(data)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--practice", action="store_true", help="expect ref-01..ref-09 instead of ep-01..ep-33")
    a = ap.parse_args(argv)
    rep = validate(a.csv, expected_endpoints=PRACTICE_ENDPOINTS if a.practice else None)
    print(rep.format())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
