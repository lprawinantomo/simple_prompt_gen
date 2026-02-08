# parser.py  (drop-in replacement)
import hashlib
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console

console = Console()

# ---------- compiled regexes (do once, use many) ----------
_ML_COMMENT = re.compile(r"/\*.*?\*/", flags=re.S)
_SL_COMMENT = re.compile(r"//[^\n]*")
_PACKAGE = re.compile(r"(?:^|\s)package\s+([^\s;]+)")
_IMPORT = re.compile(r"(?:^|\s)import\s+(?:static\s+)?([^\s;]+)")
_CLASS_DEF = re.compile(r"(?:^|\s)(?:class|interface|enum)\s+(\w+)")
_METHOD_DEF = re.compile(
    r"(?:(public|protected|private|static|abstract|final|synchronized)\s+)*"
    r"([\w<>\[\]]+)\s+(\w+)\s*\("
)

# ---------- tiny Cython accelerator (optional) ----------
try:
    from ._gscanner import scan_methods_cy  # type: ignore

    CY_AVAILABLE = True
except ImportError:
    CY_AVAILABLE = False

# ---------- helpers ----------
FileHash = str


@lru_cache(maxsize=256)
def _file_digest(path: Path) -> FileHash:
    """Unique ID for a file (path + mtime)."""
    st = path.stat()
    return f"{path}:{st.st_mtime_ns}"


def _strip_comments(source: str) -> str:
    """Remove // and /* */ in one pass."""
    source = _ML_COMMENT.sub("", source)
    source = _SL_COMMENT.sub("", source)
    return source


# ---------- state machine (pure Python fallback) ----------
def _scan_methods_py(source: str, file_path: Path) -> List[Dict]:
    """Pure-Python state machine – same logic as before, just faster."""
    lines = source.splitlines()
    methods: List[Dict] = []
    package = None
    imports: List[str] = []

    # quick metadata
    m = _PACKAGE.search(source)
    if m:
        package = m.group(1)
    for m in _IMPORT.finditer(source):
        imports.append(m.group(1))

    class_name: Optional[str] = None
    brace_depth = 0
    in_method = False
    current: Optional[Dict] = None
    method_brace = 0
    start_line = 0

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        # class / interface / enum
        m = _CLASS_DEF.search(line)
        if m and brace_depth == 0:
            class_name = m.group(1)

        # method header
        m = _METHOD_DEF.search(line)
        if m and brace_depth == 0:
            mod, ret, name = m.group(1) or "", m.group(2), m.group(3)
            if name in {"if", "for", "while", "try", "catch", "switch"}:
                continue  # false positive
            current = {
                "file": str(file_path),
                "package": package,
                "imports": imports,
                "class": class_name,
                "method": name,
                "return_type": ret,
                "modifiers": mod.split(),
                "parameters": [],  # TODO: parse param list
                "start_line": i,
                "end_line": -1,
                "source": "",
            }
            start_line = i
            in_method = True
            method_brace = brace_depth

        # braces
        brace_depth += line.count("{") - line.count("}")
        if in_method and brace_depth == method_brace:
            # method ended
            current["end_line"] = i
            current["source"] = "\n".join(lines[start_line : i + 1])
            methods.append(current)
            in_method = False
            current = None

    return methods


# ---------- choose accelerator ----------
if CY_AVAILABLE:
    scan_methods = scan_methods_cy
else:
    scan_methods = _scan_methods_py


# ---------- public API ----------
class SimpleParser:
    def __init__(self):
        self.jvm_started = False
        self.groovy_jar = None

    # ---------- entry point ----------
    def parse_file(self, file_path: Path) -> List[Dict]:
        try:
            digest = _file_digest(file_path)
            source = file_path.read_text(encoding="utf-8")
            source = _strip_comments(source)
            return scan_methods(source, file_path)
        except Exception as e:
            console.print(f"[yellow]Skip {file_path}: {e}[/yellow]")
            return []

    # ---------- bulk indexer ----------
    def index_files(
        self, files: List[Path]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, Dict]]]:
        console.print(f"[cyan]Indexing {len(files)} files …[/cyan]")
        t0 = time.time()

        method_index = defaultdict(list)
        class_methods = defaultdict(dict)
        total = 0

        # parallel parse
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            for methods in pool.map(self.parse_file, files):
                for m in methods:
                    param_types = [p["type"] for p in m.get("parameters", [])]
                    mid = f"{m['method']}({','.join(param_types)})"
                    m["method_id"] = mid
                    m["simple_signature"] = f"{m.get('return_type', 'void')} {mid}"
                    method_index[m["method"]].append(m)
                    if m["class"]:
                        class_methods[m["class"]][mid] = m
                    total += 1

        console.print(
            f"[green]Indexed {total} methods in {time.time() - t0:.2f}s[/green]"
        )
        return method_index, class_methods


# ---------- CLI quick test ----------
if __name__ == "__main__":
    parser = SimpleParser()
    roots = [Path(p) for p in sys.argv[1:]] or [Path(".")]
    groovy_files = [p for r in roots for p in r.rglob("*.groovy")]
    idx, cls = parser.index_files(groovy_files)
    console.print(f"{len(idx)} unique method names, {len(cls)} classes")
