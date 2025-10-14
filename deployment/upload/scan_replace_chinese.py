#!/usr/bin/env python3
import argparse
import os
import re
import sys
import json
from typing import Dict, List, Tuple

try:
    import yaml  # optional, only when --map is yaml/yml
except Exception:
    yaml = None

CJK_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')

DEFAULT_INCLUDE_EXTS = {
    ".py", ".md", ".markdown", ".txt", ".json", ".yml", ".yaml",
    ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash", ".zsh",
    ".js", ".ts", ".tsx", ".jsx",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".cu", ".cuh",
}

EXCLUDE_DIRS = {
    ".git", ".svn", ".hg",
    ".mypy_cache", ".pytest_cache", "__pycache__",
    "build", "dist", "node_modules",
    ".venv", "venv", "env", "ENV",
    "logs", "checkpoints", "output", "outputs", "wandb",
    ".idea", ".vscode", ".DS_Store",
}

def is_binary_file(path: str, chunk_size: int = 2048) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(chunk_size)
        if b"\x00" in chunk:
            return True
        return False
    except Exception:
        return True

def guess_comment_prefix(ext: str) -> str:
    # Not used by default; kept for future extension.
    if ext in {".py", ".sh", ".bash", ".zsh"}:
        return "# "
    if ext in {".js", ".ts", ".tsx", ".jsx", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".cu", ".cuh"}:
        return "// "
    if ext in {".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf"}:
        return "# "
    if ext in {".md", ".markdown"}:
        return ""  # markdown lines differ; avoid auto-comment
    return ""

def load_mapping(map_path: str) -> Dict[str, str]:
    if not map_path:
        return {}
    ext = os.path.splitext(map_path)[1].lower()
    with open(map_path, "r", encoding="utf-8") as f:
        text = f.read()
    if ext in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed. pip install pyyaml")
        data = yaml.safe_load(text) or {}
    elif ext == ".json":
        data = json.loads(text or "{}") or {}
    else:
        raise ValueError(f"Unsupported map file extension: {ext}")
    if not isinstance(data, dict):
        raise ValueError("Mapping file must contain a dict from zh_string to en_string.")
    # ensure keys are str
    out = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        out[k] = v
    return out

def iter_text_files(root: str, include_exts: set) -> List[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1].lower()
            if include_exts and ext not in include_exts:
                continue
            if is_binary_file(path):
                continue
            yield path

def find_cjk_lines(lines: List[str]) -> List[Tuple[int, str]]:
    hits = []
    for i, line in enumerate(lines, start=1):
        if CJK_RE.search(line):
            hits.append((i, line.rstrip("\n")))
    return hits

def replace_by_mapping(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    # Sort keys by length desc to avoid partial overshadowing
    for k in sorted(mapping.keys(), key=len, reverse=True):
        text = re.sub(re.escape(k), mapping[k], text)
    return text

def process_file(path: str, mapping: Dict[str, str], in_place: bool, backup: bool, print_lines: bool) -> Tuple[bool, int]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return (False, 0)

    lines = content.splitlines(True)
    hits = find_cjk_lines(lines)
    if not hits:
        return (False, 0)

    if print_lines:
        for lineno, line in hits:
            sys.stdout.write(f"[CJK] {path}:{lineno}: {line}\n")
    else:
        sys.stdout.write(f"[CJK] {path} ({len(hits)} lines)\n")

    if in_place:
        new_content = replace_by_mapping(content, mapping)
        if new_content != content:
            if backup:
                try:
                    with open(path + ".bak", "w", encoding="utf-8", errors="ignore") as f:
                        f.write(content)
                except Exception:
                    pass
            with open(path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(new_content)
            return (True, len(hits))
    return (False, len(hits))

def main():
    parser = argparse.ArgumentParser(
        description="Scan repository for Chinese characters and optionally replace them via mapping."
    )
    parser.add_argument("--root", type=str, default=".", help="Root directory to scan.")
    parser.add_argument("--map", type=str, default="", help="Path to zh->en mapping file (yaml/yml/json).")
    parser.add_argument("--in-place", action="store_true", help="Apply replacements in place based on mapping.")
    parser.add_argument("--backup", action="store_true", help="Write a .bak backup before in-place changes.")
    parser.add_argument("--print-lines", action="store_true", help="Print each matched line.")
    parser.add_argument("--ext", type=str, nargs="*", default=None, help="Override extensions to include, e.g. --ext .py .md")
    args = parser.parse_args()

    include_exts = set(args.ext) if args.ext else DEFAULT_INCLUDE_EXTS
    mapping = load_mapping(args.map) if args.map else {}

    total_files = 0
    changed_files = 0
    total_hits = 0

    for path in iter_text_files(args.root, include_exts):
        modified, hits = process_file(path, mapping, args.in_place, args.backup, args.print_lines)
        if hits > 0:
            total_files += 1
            total_hits += hits
        if modified:
            changed_files += 1

    sys.stdout.write(
        f"\nSummary: files_with_cjk={total_files}, total_cjk_lines={total_hits}, files_modified={changed_files}\n"
    )

if __name__ == "__main__":
    main()
