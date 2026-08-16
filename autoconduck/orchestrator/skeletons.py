"""AST skeletons, gitignore filtering, and cross-file dependency mapping for planner context."""
from __future__ import annotations

import ast
import fnmatch
import json
import re
from pathlib import Path
from typing import Any

_DEFAULT_EXCLUDES = {
    "build",
    "graphify-out",
    ".git",
    "__pycache__",
    "node_modules",
    ".autoconduck",
    "backups",
    "dist",
    ".pytest_cache",
    ".gemini",
    ".agents",
}

_IGNORED_EXTENSIONS = {
    ".lock",
    ".log",
    ".bak",
    ".map",
    ".pyc",
    ".pyo",
    ".min.js",
    ".min.css",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}


def load_gitignore_patterns(root: Path) -> list[str]:
    """Read patterns from .gitignore in the workspace root if present."""
    patterns = []
    gi_path = root / ".gitignore"
    if gi_path.is_file():
        try:
            for line in gi_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        except OSError:
            pass
    return patterns


def is_ignored_path(
    path_str: str,
    root: Path | None = None,
    patterns: list[str] | None = None,
) -> bool:
    """Return True if path matches default excludes, non-code extensions, or .gitignore rules."""
    norm_path = path_str.replace("\\", "/")
    parts = [p.lower() for p in Path(norm_path).parts]
    if any(part in _DEFAULT_EXCLUDES for part in parts):
        return True

    suffix = Path(norm_path).suffix.lower()
    if suffix in _IGNORED_EXTENSIONS:
        return True

    name = Path(norm_path).name.lower()
    if "-lock." in name or name.endswith(".lock"):
        return True

    if patterns is None:
        patterns = load_gitignore_patterns(root or Path.cwd())

    for pat in patterns:
        pat_clean = pat.rstrip("/")
        if fnmatch.fnmatch(norm_path, pat) or fnmatch.fnmatch(norm_path, f"*/{pat}"):
            return True
        if fnmatch.fnmatch(norm_path, f"{pat_clean}/*") or fnmatch.fnmatch(
            norm_path, f"*/{pat_clean}/*"
        ):
            return True
        if any(fnmatch.fnmatch(part, pat_clean) for part in parts):
            return True
    return False


def resolve_1hop_dependencies(
    candidate_files: list[str],
    root: Path | None = None,
    max_total_files: int = 6,
) -> list[str]:
    """Scan candidate files for local imports/requires to discover 1-hop local dependencies."""
    root = root or Path.cwd()
    patterns = load_gitignore_patterns(root)
    result = list(candidate_files)
    seen = {p.replace("\\", "/") for p in candidate_files}

    for path_str in list(candidate_files):
        if len(result) >= max_total_files:
            break
        full_path = root / path_str
        if not full_path.is_file():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        parent_dir = Path(path_str).parent
        # 1. Python imports
        if path_str.endswith(".py"):
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if len(result) >= max_total_files:
                        break
                    mod = None
                    if isinstance(node, ast.ImportFrom):
                        mod = node.module
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            mod = alias.name
                    if mod:
                        mod_path_rel = mod.replace(".", "/")
                        candidates = [
                            parent_dir / f"{mod_path_rel}.py",
                            parent_dir / mod_path_rel / "__init__.py",
                            Path(f"{mod_path_rel}.py"),
                            Path(f"{mod_path_rel}/__init__.py"),
                        ]
                        for cand in candidates:
                            norm = cand.as_posix()
                            if (root / cand).is_file() and norm not in seen and not is_ignored_path(norm, root, patterns):
                                result.append(norm)
                                seen.add(norm)
                                break
            except Exception:
                pass
        # 2. JS/TS imports
        elif path_str.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs")):
            for m in re.finditer(r"""(?:import|require)\s*\(?['"](\.[^'"]+)['"]""", content):
                if len(result) >= max_total_files:
                    break
                rel_mod = m.group(1)
                for ext in ["", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"]:
                    cand = (parent_dir / f"{rel_mod}{ext}").as_posix()
                    if (root / cand).is_file() and cand not in seen and not is_ignored_path(cand, root, patterns):
                        result.append(cand)
                        seen.add(cand)
                        break

    return result[:max_total_files]


def extract_python_skeleton(code: str) -> str:
    """Extract AST symbols: classes, methods, functions, types, and imports."""
    try:
        tree = ast.parse(code)
    except Exception:
        # Fallback regex for unparseable Python
        lines = []
        for line in code.splitlines():
            if re.match(r"^\s*(?:class|def|import|from)\s+", line):
                lines.append(line.rstrip())
        return "\n".join(lines[:50])

    lines: list[str] = []
    doc = ast.get_docstring(tree)
    if doc:
        first_doc = doc.strip().splitlines()[0]
        lines.append(f'"""{first_doc[:120]}"""')

    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or "."
            names = ", ".join(a.name for a in node.names[:6])
            if len(node.names) > 6:
                names += f", ... (+{len(node.names)-6} more)"
            imports.append(f"from {mod} import {names}")

    if imports:
        lines.append("Imports:\n  " + "\n  ".join(imports[:12]))
        if len(imports) > 12:
            lines.append(f"  ... (+{len(imports)-12} more imports)")

    classes: list[str] = []
    functions: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
            methods: list[str] = []
            class_doc = ast.get_docstring(node)
            if class_doc:
                methods.append(f'    """{class_doc.strip().splitlines()[0][:80]}"""')
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                    args = ast.unparse(item.args)
                    returns = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                    methods.append(f"    {prefix} {item.name}({args}){returns}")
            classes.append(header + ("\n" + "\n".join(methods[:15]) if methods else " pass"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args = ast.unparse(node.args)
            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            doc_str = ""
            fn_doc = ast.get_docstring(node)
            if fn_doc:
                doc_str = f'  # {fn_doc.strip().splitlines()[0][:70]}'
            functions.append(f"{prefix} {node.name}({args}){returns}{doc_str}")

    if classes:
        lines.append("Classes:\n  " + "\n  ".join("\n  ".join(c.splitlines()) for c in classes))
    if functions:
        lines.append("Functions:\n  " + "\n  ".join(functions))

    return "\n".join(lines)


def extract_js_ts_skeleton(content: str) -> str:
    """Extract TypeScript / JavaScript exported symbols, interfaces, and imports."""
    lines: list[str] = []
    for line in content.splitlines():
        trimmed = line.strip()
        if re.match(r"^(?:export\s+|import\s+|interface\s+|type\s+|class\s+|function\s+|const\s+\w+\s*=\s*(?:function|\([^)]*\)\s*=>))", trimmed):
            lines.append(trimmed[:120])
            if len(lines) >= 40:
                lines.append("... (+more symbols truncated)")
                break
    return "\n".join(lines) if lines else content[:500]


def extract_markdown_skeleton(content: str) -> str:
    """Extract markdown header structure."""
    lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("#"):
            lines.append(line[:100])
            if len(lines) >= 30:
                break
    return "\n".join(lines) if lines else content[:400]


def extract_json_toml_skeleton(content: str) -> str:
    """Extract top-level keys or truncated structure for structured data files."""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            keys = list(data.keys())
            return f"JSON keys ({len(keys)}): {', '.join(str(k) for k in keys[:15])}"
        if isinstance(data, list):
            return f"JSON array (length {len(data)})"
    except Exception:
        pass
    lines = content.splitlines()[:20]
    return "\n".join(lines)


def extract_file_skeleton(path: str, content: str) -> str:
    """Generate a high-fidelity, compact structural skeleton for any file type."""
    lower = path.lower()
    if lower.endswith(".py"):
        return extract_python_skeleton(content)
    if lower.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs")):
        return extract_js_ts_skeleton(content)
    if lower.endswith((".md", ".rst", ".txt")):
        return extract_markdown_skeleton(content)
    if lower.endswith((".json", ".toml", ".yaml", ".yml")):
        return extract_json_toml_skeleton(content)
    # Default fallback
    lines = content.splitlines()[:25]
    return "\n".join(lines)


def extract_dependency_map(files: dict[str, str]) -> str:
    """Build a cross-file relationship map showing references between candidate files."""
    if len(files) <= 1:
        return ""
    dep_lines: list[str] = []
    file_stems = {Path(p).stem: p for p in files.keys()}
    for path, code in files.items():
        stem = Path(path).stem
        imported: list[str] = []
        for other_stem, other_path in file_stems.items():
            if other_stem != stem and re.search(r"\b" + re.escape(other_stem) + r"\b", code):
                imported.append(other_path)
        if imported:
            dep_lines.append(f"  - {path} -> {', '.join(sorted(imported))}")
    if dep_lines:
        return "CROSS-FILE DEPENDENCY MAP:\n" + "\n".join(dep_lines)
    return ""


def format_structural_context(files: dict[str, str], ground_truth: str = "") -> str:
    """Format AST skeletons, cross-file dependency maps, and recon scout ground truth."""
    if not files and not ground_truth:
        return ""
    parts = ["\n\nFILE STRUCTURE & SKELETONS (ground truth for planning):"]
    for path, content in files.items():
        skeleton = extract_file_skeleton(path, content)
        parts.append(f"### {path}\n{skeleton}")

    dep_map = extract_dependency_map(files)
    if dep_map:
        parts.append(dep_map)

    if ground_truth:
        parts.append(f"RECON GROUND TRUTH EVIDENCE:\n{ground_truth}")

    return "\n\n".join(parts)
