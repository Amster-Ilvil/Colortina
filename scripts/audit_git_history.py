#!/usr/bin/env python3
"""Read-only privacy inspection of ALL fetched Git objects and legacy payloads.

Scans every reachable blob, not merely files currently tracked on main.
Do not print raw matches: the CI log must never disclose the secret it finds.
"""
from __future__ import annotations

import base64
import io
import lzma
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter

LOCAL_HOME_LOCATIONS: set[tuple[str, int]] = set()
LOCAL_HOME_BLOBS: set[str] = set()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_SOURCE = {"scripts/privacy_audit.py", "scripts/audit_git_history.py"}
MAX_SCAN = 12 * 1024 * 1024
MAX_ARCHIVE = 48 * 1024 * 1024
SNAPSHOTS = (
    ("098fa8f9663f4bfafea68e4a8e0ee41c0f059d17", ".github/colortina_payload/"),
    ("b133c7569f021ec2c87b5fb2c7a34ddf75fba558", "_direct_payload/"),
    ("b133c7569f021ec2c87b5fb2c7a34ddf75fba558", "source_sync/"),
    ("bb2e2848ef09", ".github/v51327-payload/"),
)
SENSITIVE_NAMES = re.compile(
    r"(?i)(?:^|/)(?:\.env(?:\.[^/]*)?|credentials?(?:\.[^/]*)?|"
    r"secrets?(?:\.[^/]*)?|service-account(?:\.[^/]*)?|"
    r"[^/]+\.(?:pem|p12|pfx|key|ccproject|ccpalette|ccscene|ccstyle))$")
CHECKPOINT_NAME = re.compile(
    r"(?i)(?:^|/)(?:generator\.zip|(?:models|weights|checkpoints)/[^/]+\."
    r"(?:pt|pth|ckpt|safetensors))$")
CREDENTIAL_PATTERNS = (
    ("private-key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("github-token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("api-token", re.compile(rb"(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|"
                             rb"sk-ant-[A-Za-z0-9_-]{20,}|"
                             rb"hf_[A-Za-z0-9]{20,}|"
                             rb"AIza[0-9A-Za-z_-]{30,}|"
                             rb"AKIA[0-9A-Z]{16})")),
    ("url-credential", re.compile(rb"https?://[^/\s:@]{1,80}:[^@\s/]{1,100}@")),
)
USER_PATHS = (
    re.compile(rb"/Users/(?!Shared/|runner/|<|USER/|username/|yourname/)[^/\s\"']{2,}"),
    re.compile(rb"/home/(?!runner/|<|USER/|username/|yourname/)[^/\s\"']{2,}"),
    re.compile(rb"(?i)[A-Z]:\\Users\\(?!Public\\|runneradmin\\|<|USER\\|username\\)[^\\\s\"']{2,}"),
)


def git(*args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=ROOT, stderr=subprocess.DEVNULL)


def inspect_data(path: str, data: bytes, categories: Counter) -> None:
    if path in ROOT_SOURCE or path.endswith("/privacy_audit.py"):
        return
    if len(data) > MAX_SCAN:
        categories["oversized-uninspected"] += 1
        return
    if b"\0" in data[:8192]:
        return
    for label, regex in CREDENTIAL_PATTERNS:
        if regex.search(data):
            categories[label] += 1
    # Patterns printed as Python regex examples are not user home folders.
    if "privacy_scan" not in path:
        for pattern in USER_PATHS:
            for match in pattern.finditer(data):
                # Legacy README examples use a literal Chinese placeholder,
                # not a real Windows user account (C:\\Users\\你的实际路径).
                if "你的实际路径" in match.group(0).decode("utf-8", "ignore"):
                    continue
                start = data.rfind(b"\n", 0, match.start()) + 1
                line = data[start:data.find(b"\n", match.end()) if b"\n" in data[match.end():] else len(data)]
                if b"re.compile" not in line and b"regex" not in line and b"pattern" not in line:
                    categories["possible-local-home"] += 1
                    line_number = data.count(b"\n", 0, match.start()) + 1
                    LOCAL_HOME_LOCATIONS.add((path, line_number))
                    break


def inspect_archive(label: str, raw: bytes, categories: Counter) -> int:
    if len(raw) > MAX_ARCHIVE:
        categories["large-encoded-archive-uninspected"] += 1
        return 0
    try:
        raw = base64.b64decode(b"".join(raw.split()), validate=True)
        if raw.startswith(b"\xfd7zXZ"):
            raw = lzma.decompress(raw, memlimit=256 * 1024 * 1024)
    except (ValueError, lzma.LZMAError, OSError, EOFError):
        categories["encoded-archive-not-decodable"] += 1
        return 0
    if len(raw) > MAX_ARCHIVE:
        categories["large-decoded-archive-uninspected"] += 1
        return 0
    scanned = 0
    try:
        if zipfile.is_zipfile(io.BytesIO(raw)):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                for item in archive.infolist():
                    if item.is_dir():
                        continue
                    if item.file_size > MAX_SCAN:
                        categories["large-archive-entry-uninspected"] += 1
                        continue
                    member = item.filename
                    if SENSITIVE_NAMES.search(member) or CHECKPOINT_NAME.search(member):
                        categories["sensitive-archive-path"] += 1
                    inspect_data(member, archive.read(item), categories)
                    scanned += 1
        else:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
                for item in archive:
                    if not item.isfile():
                        continue
                    if item.size > MAX_SCAN:
                        categories["large-archive-entry-uninspected"] += 1
                        continue
                    member = item.name.removeprefix("./")
                    if SENSITIVE_NAMES.search(member) or CHECKPOINT_NAME.search(member):
                        categories["sensitive-archive-path"] += 1
                    handle = archive.extractfile(item)
                    if handle is not None:
                        inspect_data(member, handle.read(MAX_SCAN + 1), categories)
                        scanned += 1
    except (tarfile.TarError, zipfile.BadZipFile, ValueError, OSError, EOFError):
        categories["encoded-archive-not-decodable"] += 1
    return scanned


def legacy_segments(commit: str, folder: str) -> list[str]:
    try:
        paths = git("ls-tree", "-r", "--name-only", commit, "--", folder).decode().splitlines()
    except subprocess.CalledProcessError:
        return []
    return sorted((name for name in paths if
                   re.search(r"/part-?\d+\.b64$", name)), key=lambda x:
                  int(re.search(r"(\d+)\.b64$", x).group(1)))


def main() -> int:
    categories: Counter = Counter()
    objects = {}
    for row in git("rev-list", "--objects", "--all").splitlines():
        sha, _, name = row.partition(b" ")
        if not name:
            continue
        objects.setdefault(sha.decode(), name.decode("utf-8", "replace"))
    seen_blobs = 0
    for sha, path in objects.items():
        if git("cat-file", "-t", sha).strip() != b"blob":
            continue
        seen_blobs += 1
        data = git("cat-file", "-p", sha)
        if SENSITIVE_NAMES.search(path):
            categories["sensitive-history-path"] += 1
        if CHECKPOINT_NAME.search(path):
            categories["checkpoint-in-history"] += 1
        if path.endswith(".b64"):
            categories["historical-encoded-fragment"] += 1
        else:
            previous = categories["possible-local-home"]
            inspect_data(path, data, categories)
            if categories["possible-local-home"] > previous:
                LOCAL_HOME_BLOBS.add(sha)
    archive_entries = 0
    for commit, folder in SNAPSHOTS:
        parts = legacy_segments(commit, folder)
        if not parts:
            continue
        try:
            joined = b"".join(git("show", f"{commit}:{p}").strip() for p in parts)
        except subprocess.CalledProcessError:
            categories["encoded-archive-unavailable"] += 1
            continue
        archive_entries += inspect_archive(folder, joined, categories)
    fatal_labels = (
        "private-key", "github-token", "api-token", "url-credential",
        "possible-local-home", "sensitive-history-path",
        "checkpoint-in-history", "sensitive-archive-path",
    )
    failures = {name: categories[name] for name in fatal_labels if categories[name]}
    warnings = {name: value for name, value in categories.items()
                if name not in fatal_labels}
    # Deliberately do not print secret values, matched lines or private paths.
    print(f"Git history privacy scan: {seen_blobs} reachable blobs; "
          f"{archive_entries} decoded legacy archive members.")
    print("Potential exposure categories (counts only):", failures or "none")
    # Source filename and line only; never echo the actual home directory value.
    if LOCAL_HOME_LOCATIONS:
        print("Historical home-path source locations (values redacted):",
              sorted(LOCAL_HOME_LOCATIONS)[:25])
        print("Historical source blob identifiers (no values):",
              sorted(LOCAL_HOME_BLOBS)[:25])
    print("Historical archive/coverage notes (counts only):", warnings or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
