# _gscanlr.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

from libc.stdlib cimport malloc, free
import re

# ---------- Python helpers (same regexes used in parser.py) ----------
cdef object _PACKAGE    = re.compile(r'(?:^|\s)package\s+([^\s;]+)')
cdef object _IMPORT     = re.compile(r'(?:^|\s)import\s+(?:static\s+)?([^\s;]+)')
cdef object _CLASS_DEF  = re.compile(r'(?:^|\s)(?:class|interface|enum)\s+(\w+)')
cdef object _METHOD_DEF = re.compile(
    r'(?:(public|protected|private|static|abstract|final|synchronized)\s+)*'
    r'([\w<>\[\]]+)\s+(\w+)\s*\('
)

# ---------- fast strip-comments -------------------------------------------------
cdef inline str _strip_comments(str source):
    source = re.sub(r'//.*', '', source)
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    return source

# ---------- main entry point ----------------------------------------------------
def scan_methods_cy(str source, file_path):
    """
    Cython implementation of the state-machine.
    Returns list[dict] with identical keys as the Python version.
    """
    cdef int i, brace_depth = 0, method_brace = 0, start_line = 0
    cdef str line, stripped, class_name = None, package = None
    cdef list lines = source.splitlines()
    cdef list imports = []
    cdef list methods = []
    cdef dict current = None
    cdef bint in_method = False

    # quick metadata
    m = _PACKAGE.search(source)
    if m:
        package = m.group(1)
    for m in _IMPORT.finditer(source):
        imports.append(m.group(1))

    for i in range(len(lines)):
        line = lines[i]
        stripped = line.strip()

        # skip empty / single-line comments
        if not stripped or stripped.startswith('//'):
            continue

        # skip multi-line comments quickly
        if '/*' in stripped:
            if '*/' not in stripped:
                i += 1
                while i < len(lines) and '*/' not in lines[i]:
                    i += 1
                if i < len(lines):
                    i += 1
                continue
            else:
                # one-liner /* ... */
                continue

        # class / interface / enum
        m = _CLASS_DEF.search(line)
        if m and brace_depth == 0:
            class_name = m.group(1)

        # method header
        m = _METHOD_DEF.search(line)
        if m and brace_depth == 0:
            mod, ret, name = m.group(1) or '', m.group(2), m.group(3)
            if name in {'if', 'for', 'while', 'try', 'catch', 'switch'}:
                continue  # false positive
            start_line = i
            method_brace = brace_depth
            in_method = True
            current = {
                'file': str(file_path),
                'package': package,
                'imports': imports.copy(),
                'class': class_name,
                'method': name,
                'return_type': ret,
                'modifiers': mod.split(),
                'parameters': [],
                'start_line': start_line,
                'end_line': -1,
                'source': '',
            }

        # braces
        brace_depth += line.count('{') - line.count('}')
        if in_method and brace_depth == method_brace:
            # method ended
            current['end_line'] = i
            current['source'] = '\n'.join(lines[start_line:i+1])
            methods.append(current)
            in_method = False
            current = None

    return methods
