"""AST & symbol extractor for selective repository indexing."""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Sequence

from autoconduck.knowledge.models import CodeChunk

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".agents",
    "dist",
    "build",
    "node_modules",
}


def _embed_text(text: str, dim: int = 16) -> list[float]:
    """Generate a deterministic normalized term hash vector for text."""
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    words = text.lower().split()
    for word in words:
        h = int(hashlib.md5(word.encode("utf-8", errors="ignore")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0

    # Normalize vector
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [round(x / norm, 6) for x in vec]
    return vec


def extract_repo_symbols(root_dir: str | Path, max_files: int = 50) -> list[CodeChunk]:
    """Scan Python files and extract functions, classes, and module contracts."""
    root = Path(root_dir).resolve()
    chunks: list[CodeChunk] = []

    if not root.exists():
        return chunks

    file_count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]

        for fname in filenames:
            if not fname.endswith(".py"):
                continue

            file_path = Path(dirpath) / fname
            rel_path = str(file_path.relative_to(root)).replace("\\", "/")

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            try:
                tree = ast.parse(content, filename=rel_path)
            except SyntaxError:
                continue

            # Module docstring chunk
            docstring = ast.get_docstring(tree) or ""
            if docstring:
                chunk_id = f"{rel_path}:doc"
                chunks.append(
                    CodeChunk(
                        id=chunk_id,
                        symbol=fname,
                        file=rel_path,
                        content=f"# {rel_path}\n{docstring}",
                        vector=_embed_text(f"{fname} {docstring}"),
                        metadata={"type": "module_doc"},
                    )
                )

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = f"def {node.name}(...)"
                    fn_doc = ast.get_docstring(node) or ""
                    fn_lines = content.splitlines()[node.lineno - 1 : min(node.end_lineno or node.lineno + 15, node.lineno + 25)]
                    fn_snippet = "\n".join(fn_lines)
                    chunk_id = f"{rel_path}:{node.name}:{node.lineno}"
                    chunks.append(
                        CodeChunk(
                            id=chunk_id,
                            symbol=node.name,
                            file=rel_path,
                            content=fn_snippet,
                            vector=_embed_text(f"{node.name} {fn_doc} {fn_snippet}"),
                            metadata={"type": "function", "lineno": node.lineno},
                        )
                    )
                elif isinstance(node, ast.ClassDef):
                    cls_doc = ast.get_docstring(node) or ""
                    cls_lines = content.splitlines()[node.lineno - 1 : min(node.end_lineno or node.lineno + 20, node.lineno + 30)]
                    cls_snippet = "\n".join(cls_lines)
                    chunk_id = f"{rel_path}:{node.name}:{node.lineno}"
                    chunks.append(
                        CodeChunk(
                            id=chunk_id,
                            symbol=node.name,
                            file=rel_path,
                            content=cls_snippet,
                            vector=_embed_text(f"{node.name} {cls_doc} {cls_snippet}"),
                            metadata={"type": "class", "lineno": node.lineno},
                        )
                    )

            file_count += 1
            if file_count >= max_files:
                break

        if file_count >= max_files:
            break

    return chunks
