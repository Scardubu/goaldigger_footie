"""HTTP Timeout Audit Script

Scans codebase for aiohttp / requests session creations and highlights
places where explicit timeout configuration is missing or ambiguous.
This is a static heuristic pass; not a guarantee of runtime correctness.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_EXT = {'.py'}

AIOHTTP_SESSION_REGEX = re.compile(r"aiohttp\.ClientSession\((?P<args>[^)]*)\)")
REQUESTS_SESSION_REGEX = re.compile(r"requests\.Session\(")

def scan_file(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore')
    findings = []
    for m in AIOHTTP_SESSION_REGEX.finditer(text):
        args = m.group('args')
        if 'timeout=' not in args:
            findings.append((path, m.start(), 'aiohttp', 'NO_TIMEOUT_ARG'))
    for m in REQUESTS_SESSION_REGEX.finditer(text):
        # For requests.Session we rely on later usage; just record presence
        findings.append((path, m.start(), 'requests', 'SESSION_CREATE'))
    return findings

def main():
    results = []
    for root, _dirs, files in os.walk(ROOT):
        for f in files:
            if Path(f).suffix in TARGET_EXT:
                p = Path(root) / f
                try:
                    results.extend(scan_file(p))
                except Exception:
                    pass
    print("HTTP Timeout Audit Report")
    print("Type | Status | File:Line")
    for path, pos, typ, status in results:
        # Approximate line number
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                data = fh.read()
            line_no = data.count('\n', 0, pos) + 1
        except Exception:
            line_no = '?'
        print(f"{typ} | {status} | {path.relative_to(ROOT)}:{line_no}")
    missing = [r for r in results if r[2] == 'aiohttp' and r[3] == 'NO_TIMEOUT_ARG']
    print("\nSummary:")
    print(f"Total aiohttp sessions without explicit timeout arg: {len(missing)}")
    print("(Note: Some wrappers may inject defaults; verify manually where appropriate.)")

if __name__ == "__main__":
    main()
