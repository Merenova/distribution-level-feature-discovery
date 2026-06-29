#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXCLUDED_PREFIXES = (
    ".git/",
    ".venv/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    "archive/",
    "data/",
    "logs/",
    "tmp/",
)

EXCLUDED_SUFFIXES = (
    ".arrow",
    ".bin",
    ".ckpt",
    ".gif",
    ".jpg",
    ".jpeg",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".safetensors",
    ".sqlite",
    ".zip",
)

def _literal(*parts: str) -> re.Pattern[str]:
    return re.compile(re.escape("".join(parts)))


def _word(*parts: str, ignore_case: bool = False) -> re.Pattern[str]:
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(r"\b" + "".join(parts) + r"\b", flags)


_REVIEW_TERM = "rebut" + "tal"
_LOCAL_USER = "/home/" + "hyunjin"
_REPO_NAME = "latent" + "_planning"
_REMOTE_OWNER = "Mere" + "nova"
_ROOT_USER = "ro" + "ot"
_PUBLIC_IPV4_PATTERN = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")

BANNED_FILENAME_PATTERNS = [
    re.compile(re.escape(_REVIEW_TERM), re.IGNORECASE),
    re.compile(r"author\s+response", re.IGNORECASE),
    re.compile(re.escape("Open" + "Review"), re.IGNORECASE),
    re.compile(re.escape("IC" + "ML"), re.IGNORECASE),
    _literal("home/" + "hyunjin"),
    _literal("workspace/" + _REPO_NAME),
    _literal("root/" + _REPO_NAME),
    re.compile(re.escape(_REMOTE_OWNER), re.IGNORECASE),
    re.compile(r"(?:--?)?" + "api" + r"-key"),
    _literal(_ROOT_USER + "@"),
]

BANNED_TEXT_PATTERNS = [
    _word(_REVIEW_TERM, ignore_case=True),
    re.compile(r"\bauthor\s+response\b", re.IGNORECASE),
    _word("Open" + "Review"),
    re.compile(re.escape("IC" + "ML"), re.IGNORECASE),
    _literal(_LOCAL_USER),
    _literal("~/" + _REPO_NAME),
    _literal("/workspace/" + _REPO_NAME),
    _literal("/root/" + _REPO_NAME),
    _literal("https://github.com/" + _REMOTE_OWNER + "/"),
    _literal("git@github.com:" + _REMOTE_OWNER + "/"),
    re.compile(r"(?:--?)?" + "api" + r"-key"),
    _literal(_ROOT_USER + "@"),
    re.compile(r"\$\{[A-Za-z0-9_]+:-" + _ROOT_USER + r"\}"),
    re.compile(r"=\s*[\"']" + _ROOT_USER + r"[\"']"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9_\-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
]


def _is_public_ipv4(value: str) -> bool:
    parts = value.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return False
    if any(num < 0 or num > 255 for num in nums):
        return False
    first, second = nums[0], nums[1]
    return not (
        first == 0
        or first == 10
        or first == 127
        or (first == 172 and 16 <= second <= 31)
        or (first == 192 and second == 168)
    )


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--modified", "--others", "--exclude-standard"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def is_scannable(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("docs/superpowers/plans/"):
        return False
    if any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    if any(rel.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return False
    if "/__pycache__/" in rel or rel.endswith(".pyc"):
        return False
    return True


def text_violations(path: Path) -> list[str]:
    rel = path.relative_to(ROOT).as_posix()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    violations: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern in BANNED_TEXT_PATTERNS:
            if pattern.search(line):
                violations.append(f"{rel}:{line_no}: banned text matches {pattern.pattern!r}")
        if path.name != "uv.lock":
            for match in _PUBLIC_IPV4_PATTERN.finditer(line):
                if _is_public_ipv4(match.group(0)):
                    violations.append(f"{rel}:{line_no}: hardcoded public IPv4 address {match.group(0)!r}")
    return violations


def filename_violations(path: Path) -> list[str]:
    rel = path.relative_to(ROOT).as_posix()
    violations = [
        f"{rel}: banned filename matches {pattern.pattern!r}"
        for pattern in BANNED_FILENAME_PATTERNS
        if pattern.search(rel)
    ]
    for match in _PUBLIC_IPV4_PATTERN.finditer(rel):
        if _is_public_ipv4(match.group(0)):
            violations.append(f"{rel}: hardcoded public IPv4 address {match.group(0)!r}")
    return violations


def collect_violations() -> list[str]:
    violations: list[str] = []
    for path in candidate_files():
        if not path.exists():
            continue
        if not is_scannable(path):
            continue
        violations.extend(filename_violations(path))
        violations.extend(text_violations(path))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Check publication hygiene for source files")
    parser.add_argument("--max-lines", type=int, default=200)
    args = parser.parse_args()

    violations = collect_violations()
    if not violations:
        print("Publication hygiene check passed")
        return 0

    print("Publication hygiene violations:")
    for violation in violations[: args.max_lines]:
        print(f"  {violation}")
    if len(violations) > args.max_lines:
        print(f"  ... {len(violations) - args.max_lines} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
