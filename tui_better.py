#!/usr/bin/env python3
"""
Grails AI Prompt Compiler
Created by Lalang Prawinantomo
MIT License
"""

import argparse
import sys
import os
import re
import subprocess
import tempfile
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Dict, Set, Optional, Tuple, Any
import pyperclip

# Try to import JPype
try:
    import jpype
    import jpype.imports
    from jpype.types import *
    JPYPE_AVAILABLE = True
except ImportError:
    JPYPE_AVAILABLE = False
    console = None  # Will be initialized later

from fuzzywuzzy import process, fuzz
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import CompleteStyle
from jinja2 import Environment, FileSystemLoader, StrictUndefined
# Initialize Rich console
console = Console()

# Global state
CONTEXT_FILES: List[Path] = []
SELECTED_METHOD: Optional[str] = None
SELECTED_METHOD_INFO: Optional[Dict] = None
METHOD_INDEX: Dict = defaultdict(list)
CLASS_METHODS: Dict = defaultdict(dict)
JVM_STARTED = False
TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"

env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)

# -------------------------
# Groovy AST Parser using JPype
# -------------------------

class GroovyASTParser:
    """Parse Groovy files using Groovy's actual parser via JPype."""
    
    def __init__(self):
        self.jvm_started = False
        self.groovy_jar = None
    
    def parse_file(self, file_path: Path) -> List[Dict]:
        """Parse a Groovy file using the best available method."""
        try:
            source = file_path.read_text(encoding='utf-8')
            return self.parse_with_state_machine(source, file_path)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not parse {file_path}: {e}[/yellow]")
            return []
    
    def parse_with_jpype(self, source: str, file_path: Path) -> List[Dict]:
        """Parse using Groovy's actual parser via JPype."""
         # Fall back to regex
        return self.parse_with_state_machine(source, file_path)

    def parse_with_state_machine(self, source: str, file_path: Path) -> List[Dict]:
        """Parse Groovy using a simple state machine."""
        methods = []
        lines = source.split('\n')
        
        class_name = None
        brace_depth = 0
        in_class = False
        in_method = False
        method_start = 0
        method_signature = ""
        current_method = None
        method_brace_depth = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            stripped = line.strip()
            
            # Skip comments
            if stripped.startswith('//'):
                i += 1
                continue
            
            # Handle multiline comments
            if '/*' in line:
                if '*/' not in line:
                    # Skip to end of comment
                    while i < len(lines) and '*/' not in lines[i]:
                        i += 1
                i += 1
                continue
            
            # Track braces for scope
            open_braces = line.count('{')
            close_braces = line.count('}')
            
            # Update brace depth
            brace_depth += open_braces - close_braces
            
            # Detect class definition
            if not in_class and re.search(r'^\s*class\s+\w+', stripped):
                match = re.search(r'class\s+(\w+)', stripped)
                if match:
                    class_name = match.group(1)
                    in_class = True
                    brace_depth = 0  # Reset for class scope
            
            # Detect method start (when we're inside a class)
            if in_class and class_name and not in_method:
                # Look for method signatures
                method_match = self.detect_method_signature(stripped)
                if method_match:
                    return_type, method_name, params_str = method_match
                    
                    # Start tracking this method
                    in_method = True
                    method_start = i
                    method_signature = line
                    current_method = {
                        "method": method_name,
                        "class": class_name,
                        "file": file_path,
                        "signature": line,
                        "return_type": return_type,
                        "parameters": self.parse_parameters(params_str),
                        "start_line": i
                    }
                    method_brace_depth = brace_depth
                    
                    # Check if method has body on same line
                    if '{' in line:
                        # Method body starts immediately
                        pass
            
            # Track method body
            if in_method and current_method:
                # Check if we've left the method scope
                if brace_depth < method_brace_depth:
                    # Method ended
                    method_end = i
                    method_source = '\n'.join(lines[method_start:method_end + 1])
                    current_method["source"] = method_source
                    current_method["end_line"] = method_end
                    methods.append(current_method)
                    
                    # Reset for next method
                    in_method = False
                    current_method = None
                    method_signature = ""
            
            # Check if we've left the class
            if in_class and brace_depth < 0:
                in_class = False
                class_name = None
            
            i += 1
        
        # Handle method that goes to end of file
        if in_method and current_method:
            method_source = '\n'.join(lines[method_start:])
            current_method["source"] = method_source
            current_method["end_line"] = len(lines) - 1
            methods.append(current_method)
        
        return methods

    def detect_method_signature(self, line: str) -> Optional[Tuple[str, str, str]]:
        """Detect if a line is a method signature."""
        # Clean the line
        line = line.strip()
        
        # Skip if it's obviously not a method (keywords, etc.)
        not_method_patterns = [
            r'^\s*(if|for|while|switch|try|catch|finally|else|do|return|throw|import|package)',
            r'^\s*[{};]',
            r'^\s*$',
        ]
        
        for pattern in not_method_patterns:
            if re.match(pattern, line):
                return None
        
        # Method pattern
        method_pattern = r'^(?:@\w+(?:\([^)]*\))?\s+)*' \
                        r'(?:public\s+|private\s+|protected\s+|static\s+|synchronized\s+|final\s+|abstract\s+)*' \
                        r'(?:def\s+|([\w.<>]+)\s+)?' \
                        r'(\w+)\s*\(([^)]*)\)'
        
        match = re.match(method_pattern, line)
        if match:
            return_type = match.group(1) or "def"
            method_name = match.group(2)
            params_str = match.group(3)
            
            # Additional validation
            if method_name in ['class', 'interface', 'enum', 'trait']:
                return None
            
            return return_type, method_name, params_str
        
        return None
    
    def extract_method_body(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract method body handling braces and strings."""
        content = []
        brace_count = 0
        in_string = False
        string_char = None
        escape_next = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            
            # Track string literals to avoid counting braces in strings
            for char in line:
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                elif in_string:
                    if char == string_char:
                        in_string = False
                        string_char = None
                else:
                    if char in ['"', "'"]:
                        in_string = True
                        string_char = char
                    elif char == '{' and not in_string:
                        brace_count += 1
                    elif char == '}' and not in_string:
                        brace_count -= 1
            
            content.append(line)
            
            # End of method when we've seen braces and they're balanced
            if brace_count == 0 and '{' in line:
                return '\n'.join(content), i
        
        return '\n'.join(content), len(lines) - 1
    
    def parse_parameters(self, params_str: str) -> List[Dict]:
        """Parse parameter string into structured data."""
        params = []
        if not params_str.strip():
            return params
        
        # Split by comma, being careful with generics
        param_parts = []
        current = ""
        depth = 0  # For tracking generics <>
        
        for char in params_str:
            if char == '<':
                depth += 1
            elif char == '>':
                depth -= 1
            elif char == ',' and depth == 0:
                param_parts.append(current.strip())
                current = ""
                continue
            current += char
        
        if current.strip():
            param_parts.append(current.strip())
        
        # Parse each parameter
        for part in param_parts:
            part = part.strip()
            if not part:
                continue
            
            # Handle default values
            if '=' in part:
                part = part.split('=', 1)[0].strip()
            
            # Pattern: [final] Type name
            param_match = re.match(
                r'^(?:final\s+)?(?:def|([\w.<>\[\]\s]+?))\s+(\w+)$',
                part
            )
            
            if param_match:
                param_type = param_match.group(1).strip() if param_match.group(1) else "def"
                param_name = param_match.group(2)
                params.append({
                    "type": param_type,
                    "name": param_name
                })
            else:
                # Fallback
                words = part.split()
                if len(words) >= 2:
                    params.append({
                        "type": words[0],
                        "name": words[-1]
                    })
        
        return params

# -------------------------
# Initialize Parser
# -------------------------

PARSER = GroovyASTParser()

# -------------------------
# Indexing and Call Graph
# -------------------------

def index_files(files: List[Path]) -> Tuple[Dict, Dict]:
    """Index Groovy files."""
    console.print(f"[cyan]Indexing {len(files)} files...[/cyan]")
    
    method_index = defaultdict(list)
    class_methods = defaultdict(dict)
    total_methods = 0
    
    for file in files:
        methods = PARSER.parse_file(file)
        
        for method_info in methods:
            method_name = method_info["method"]
            params = method_info.get("parameters", [])
            param_types = [p["type"] for p in params]
            method_id = f"{method_name}({', '.join(param_types)})"
            
            enhanced_info = {
                **method_info,
                "method_id": method_id,
                "simple_signature": f"{method_info.get('return_type', 'void')} {method_name}({', '.join(param_types)})"
            }
            
            method_index[method_name].append(enhanced_info)
            class_methods[method_info["class"]][method_id] = enhanced_info
            total_methods += 1
    
    console.print(f"[green]Indexed {total_methods} methods from {len(files)} files[/green]")
    return method_index, class_methods

def extract_method_calls_with_context(source: str, current_class: str, 
                                     class_methods: Dict) -> List[Tuple[str, str]]:
    """Extract method calls using class context."""
    calls = []
    
    # Clean source
    cleaned = re.sub(r'"[^"]*"', '""', source)
    cleaned = re.sub(r"'[^']*'", "''", cleaned)
    cleaned = re.sub(r'//.*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Get all method names in current class
    current_class_methods = set()
    if current_class in class_methods:
        current_class_methods = {m["method"] for m in class_methods[current_class].values()}
    
    # Pattern for method calls
    # 1. receiver.method()
    # 2. method() (could be same-class)
    pattern = r'\b(\w+)(?:\.(\w+))?\s*\('
    
    for match in re.finditer(pattern, cleaned):
        first = match.group(1)
        second = match.group(2)
        
        if second:
            # Explicit receiver
            receiver = first
            method = second
        else:
            # No explicit receiver - check if it's a known method in this class
            method = first
            
            # Skip keywords
            keywords = {'if', 'for', 'while', 'switch', 'try', 'catch', 'finally',
                       'return', 'throw', 'new', 'assert', 'break', 'continue'}
            if method in keywords:
                continue
            
            # Check if it's a method in current class
            if method in current_class_methods:
                receiver = "this"
            else:
                # Might be a static import or local variable
                # For now, skip it to avoid false positives
                continue
        
        # Skip system calls
        if receiver in ['System', 'out', 'err', 'in', 'println', 'print']:
            continue
        
        calls.append((receiver, method))
    
    return calls

def build_symbol_table(method_info: Dict) -> Dict[str, str]:
    """Build symbol table for a method."""
    symbols = {}
    
    # Add parameters
    for param in method_info.get("parameters", []):
        symbols[param["name"]] = param["type"]
    
    # Extract local variables (simplified)
    source = method_info.get("source", "")
    
    # Remove comments for variable detection
    source_no_comments = re.sub(r'//.*', '', source)
    source_no_comments = re.sub(r'/\*.*?\*/', '', source_no_comments, flags=re.DOTALL)
    
    # Pattern for variable declarations
    var_patterns = [
        r'(?:def|([\w.<>]+))\s+(\w+)\s*=',
        r'([\w.<>]+)\s+(\w+)\s*;'
    ]
    
    for pattern in var_patterns:
        matches = re.finditer(pattern, source_no_comments)
        for match in matches:
            var_type = match.group(1) if match.group(1) else "def"
            var_name = match.group(2)
            symbols[var_name] = var_type
    
    return symbols

def resolve_receiver_call(receiver: str, method: str, symbols: Dict, class_methods: Dict) -> Optional[Dict]:
    """Resolve a method call."""
    receiver_type = symbols.get(receiver)
    
    # Handle 'this' calls
    if receiver == "this" or not receiver_type:
        # Search all classes for the method
        for class_name, methods in class_methods.items():
            for method_id, method_info in methods.items():
                if method_info["method"] == method:
                    return method_info
        return None
    
    # Search in receiver type's class
    if receiver_type in class_methods:
        methods_in_class = class_methods[receiver_type]
        for method_id, method_info in methods_in_class.items():
            if method_info["method"] == method:
                return method_info
    
    return None

def expand_call_graph(entry_method: Dict, method_index: Dict, 
                     class_methods: Dict, max_depth: int = 2) -> Dict:
    """Build call graph with improved same-class call detection."""
    queue = deque([(entry_method, 0)])
    visited = set()
    collected = {}
    callers = defaultdict(set)
    
    while queue:
        current, depth = queue.popleft()
        key = (current["file"], current["method_id"])
        
        if key in visited or depth > max_depth:
            continue
        
        visited.add(key)
        collected[key] = {
            **current,
            "depth": depth,
            "called_by": set()
        }
        
        # Extract calls WITH context
        current_class = current["class"]
        calls = extract_method_calls_with_context(
            current.get("source", ""),
            current_class,
            class_methods
        )
        
        symbols = build_symbol_table(current)
        
        for receiver, method in calls:
            # Special handling for "this" receiver
            if receiver == "this":
                # Look for method in current class
                if current_class in class_methods:
                    for method_id, method_info in class_methods[current_class].items():
                        if method_info["method"] == method:
                            resolved_key = (method_info["file"], method_info["method_id"])
                            callers[resolved_key].add(current["method"])
                            queue.append((method_info, depth + 1))
                            break
            else:
                # Regular receiver resolution
                resolved = resolve_receiver_call(receiver, method, symbols, class_methods)
                if resolved:
                    resolved_key = (resolved["file"], resolved["method_id"])
                    callers[resolved_key].add(current["method"])
                    queue.append((resolved, depth + 1))
    
    # Update caller relationships
    for key, c in callers.items():
        if key in collected:
            collected[key]["called_by"] = c
    
    console.print(f"[cyan]Call graph expanded: {len(collected)} methods[/cyan]")
    return collected
# -------------------------
# TUI Commands (Same as before)
# -------------------------

def fuzzy_find_files(pattern: str) -> List[Path]:
    """Fuzzy find Groovy files (cross-platform)."""
    if not pattern.strip():
        return []

    all_files = list(Path.cwd().rglob("*.groovy"))

    matched = process.extract(
        pattern,
        [str(f) for f in all_files],
        limit=20
    )

    return [Path(f) for f, score in matched if score > 50]

def command_add(args: List[str]) -> None:
    """Add files to context."""
    global CONTEXT_FILES, METHOD_INDEX, CLASS_METHODS
    
    if not args:
        console.print("[yellow]Usage: /add <pattern>[/yellow]")
        return
    
    pattern = " ".join(args)
    matched_files = fuzzy_find_files(pattern)
    
    if not matched_files:
        console.print(f"[yellow]No files found matching '{pattern}'[/yellow]")
        return
    
    table = Table(title="Matching Files")
    table.add_column("#", style="cyan")
    table.add_column("File Path", style="green")
    
    for i, file in enumerate(matched_files, 1):
        table.add_row(str(i), str(file))
    
    console.print(table)
    
    selection = Prompt.ask(
        "Enter numbers to add (comma-separated, 'a' for all, 'c' to cancel)",
        default="c"
    )
    
    if selection.lower() == 'c':
        return
    elif selection.lower() == 'a':
        files_to_add = matched_files
    else:
        try:
            indices = [int(i.strip()) for i in selection.split(',')]
            files_to_add = [matched_files[i-1] for i in indices if 1 <= i <= len(matched_files)]
        except:
            console.print("[red]Invalid selection[/red]")
            return
    
    added = []
    for file in files_to_add:
        if file not in CONTEXT_FILES:
            CONTEXT_FILES.append(file)
            added.append(file)
    
    if added:
        console.print(f"[green]Added {len(added)} files[/green]")
        METHOD_INDEX, CLASS_METHODS = index_files(CONTEXT_FILES)

def command_drop() -> None:
    """Drop all files from context."""
    global CONTEXT_FILES, METHOD_INDEX, CLASS_METHODS, SELECTED_METHOD, SELECTED_METHOD_INFO
    
    if not CONTEXT_FILES:
        console.print("[yellow]No files in context[/yellow]")
        return
    
    if Confirm.ask(f"Drop all {len(CONTEXT_FILES)} files from context?"):
        CONTEXT_FILES.clear()
        METHOD_INDEX.clear()
        CLASS_METHODS.clear()
        SELECTED_METHOD = None
        SELECTED_METHOD_INFO = None
        console.print("[green]Context cleared[/green]")

def display_method_signature(method_info: Dict) -> None:
    """Display method signature."""
    console.print(f"  Class: {method_info['class']}")
    console.print(f"  File: {method_info['file'].name}")
    
    params = method_info.get("parameters", [])
    param_str = ", ".join([f"{p['type']} {p['name']}" for p in params])
    return_type = method_info.get("return_type", "void")
    
    console.print(f"  Signature: {return_type} {method_info['method']}({param_str})")

def handle_overloaded_methods(method_name: str, candidates: List[Dict]) -> None:
    """Handle overloaded method selection."""
    global SELECTED_METHOD, SELECTED_METHOD_INFO
    
    table = Table(title=f"Overloaded Methods: {method_name}")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Class", style="green")
    table.add_column("Signature", style="yellow")
    
    for i, method_info in enumerate(candidates, 1):
        params = method_info.get("parameters", [])
        param_types = [p["type"] for p in params]
        return_type = method_info.get("return_type", "void")
        sig = f"{return_type}({', '.join(param_types)})"
        
        table.add_row(str(i), method_info["class"], sig[:50])
    
    console.print(table)
    
    choices = [str(i) for i in range(1, len(candidates) + 1)] + ["c"]
    
    while True:
        choice = Prompt.ask(
            f"Select method (1-{len(candidates)}, 'c' to cancel)",
            choices=choices,
            show_choices=False
        )
        
        if choice.lower() == 'c':
            console.print("[yellow]Selection cancelled[/yellow]")
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                SELECTED_METHOD = method_name
                SELECTED_METHOD_INFO = candidates[idx]
                console.print(f"[green]Selected method #{choice}:[/green]")
                display_method_signature(SELECTED_METHOD_INFO)
                return
        except ValueError:
            console.print(f"[red]Invalid input: {choice}[/red]")

def command_method(args: List[str]) -> None:
    """Select a method."""
    global SELECTED_METHOD, SELECTED_METHOD_INFO, METHOD_INDEX
    
    if not CONTEXT_FILES:
        console.print("[red]No files in context[/red]")
        return
    
    if not METHOD_INDEX:
        METHOD_INDEX, CLASS_METHODS = index_files(CONTEXT_FILES)
    
    method_names = list(METHOD_INDEX.keys())
    
    if args:
        input_str = " ".join(args)
        signature_pattern = r'^(\w+)\s*(?:\(([^)]*)\))?$'
        match = re.match(signature_pattern, input_str)
        
        if match:
            method_name = match.group(1)
            params_str = match.group(2)
            
            if params_str:
                param_types = [pt.strip() for pt in params_str.split(',') if pt.strip()]
                candidates = METHOD_INDEX.get(method_name, [])
                for candidate in candidates:
                    candidate_params = [p["type"] for p in candidate.get("parameters", [])]
                    if candidate_params == param_types:
                        SELECTED_METHOD = method_name
                        SELECTED_METHOD_INFO = candidate
                        console.print(f"[green]Found exact match:[/green]")
                        display_method_signature(candidate)
                        return
                console.print(f"[yellow]No exact match for {method_name}({params_str})[/yellow]")
            
            candidates = METHOD_INDEX.get(method_name, [])
        else:
            method_name = input_str
            candidates = METHOD_INDEX.get(method_name, [])
    else:
        try:
            completer = WordCompleter(method_names, ignore_case=True)
            method_name = pt_prompt(
                "Method name: ",
                completer=completer,
                complete_style=CompleteStyle.MULTI_COLUMN
            )
            if not method_name:
                return
            candidates = METHOD_INDEX.get(method_name, [])
        except KeyboardInterrupt:
            return
    
    if not candidates:
        matches = process.extract(method_name, method_names, limit=5)
        console.print(f"[yellow]Method '{method_name}' not found. Did you mean:[/yellow]")
        for match, score in matches:
            console.print(f"  - {match} ({score}% match)")
        return
    
    if len(candidates) == 1:
        SELECTED_METHOD = method_name
        SELECTED_METHOD_INFO = candidates[0]
        console.print(f"[green]Selected method: {method_name}[/green]")
        display_method_signature(SELECTED_METHOD_INFO)
    else:
        handle_overloaded_methods(method_name, candidates)

# -------------------------
# Prompt Generation
# -------------------------

def trim_context(method_info: Dict, max_lines: int = 50) -> str:
    """Trim method context if too long."""
    source = method_info.get("source", "")
    lines = source.split('\n')
    
    if len(lines) <= max_lines:
        return source
    
    keep_first = max_lines // 2
    keep_last = max_lines - keep_first
    
    trimmed = lines[:keep_first] + ["\n    // ... truncated ...\n"] + lines[-keep_last:]
    return '\n'.join(trimmed)

def generate_prompt(task_type: str, user_prompt: str, call_depth: int = 2) -> Optional[str]:
    """Generate AI prompt."""
    if not SELECTED_METHOD_INFO:
        console.print("[red]No method selected[/red]")
        return None
    
    try:
        # Build call graph
        graph = expand_call_graph(SELECTED_METHOD_INFO, METHOD_INDEX, CLASS_METHODS, call_depth)
        
        # Filter to most relevant methods (max 8)
        relevant_methods = {}
        for key, method in graph.items():
            if method["depth"] <= 2:  # Only include depth 0-2
                relevant_methods[key] = method
            if len(relevant_methods) >= 8:
                break
        
        # Build context
        context_lines = ["// Call Graph:"]
        for method in relevant_methods.values():
            #callers = ", ".join(method["called_by"]) if method["called_by"] else "entry"
            y = [i for i in method["called_by"] if not i == method['method']]
            indent = "  " * method["depth"]
            if y :
                callers = ", ".join(y)
                context_lines.append(f"// {indent}{method['method']} (depth {method['depth']}, called by: {callers})")   
            else :
                context_lines.append(f"// {indent}{method['method']} (depth {method['depth']}, entry method)")
        context_lines.append("")
        
        # Add method code
        context_lines.append("// Relevant Code:")
        for method in relevant_methods.values():
            if method["depth"] == 0 :
                # Full method for entry point
                context_lines.append(f"// === {method['class']}.{method['method']} (ENTRY METHOD) ===")
                context_lines.append(trim_context(method, 60))
            else:
                # Signature only for called methods
                params = method.get("parameters", [])
                param_str = ", ".join([f"{p['type']} {p['name']}" for p in params])
                return_type = method.get("return_type", "void")
                context_lines.append(f"// === {method['class']}.{method['method']} (CALLEE - READ ONLY) ===")
                context_lines.append(f"{return_type} {method['method']}({param_str}) {{ ... }}")
            context_lines.append("")
        
        context_text = "\n".join(context_lines)
        
        # Task-specific templates
        template_name = f"{task_type}.j2"

        if not (TEMPLATE_DIR / template_name).exists():
            template_name = "refactor.j2"

        template = env.get_template(template_name)

        return template.render(
            context=context_text,
            user_prompt=user_prompt or "",
        )
        
    except Exception as e:
        console.print(f"[red]Error generating prompt: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None

def command_refactor(args: List[str]) -> None:
    """Refactor command."""
    
    user_prompt = " ".join(args).strip() or None
    call_depth = Prompt.ask("Call graph depth", default="2")
    
    try:
        call_depth = int(call_depth)
    except:
        call_depth = 2
    
    prompt = generate_prompt("refactor", user_prompt, call_depth)
    if prompt:
        pyperclip.copy(prompt)
        console.print("[green]✓ Prompt copied to clipboard[/green]")
        console.print(f"\n[dim]Length: {len(prompt)} chars[/dim]")
        
        if Confirm.ask("Show preview?"):
            review_code(prompt)
            #console.print("\n[bold cyan]Preview:[/bold cyan]")
            #console.print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

def command_implement(args: List[str]) -> None:
    """Implement command."""
    if not args:
        console.print("[yellow]Usage: /implement <prompt>[/yellow]")
        return
    
    user_prompt = " ".join(args)
    prompt = generate_prompt("implement", user_prompt)
    if prompt:
        pyperclip.copy(prompt)
        console.print("[green]✓ Prompt copied to clipboard[/green]")

def command_ask(args: List[str]) -> None:
    """Ask command."""
    if not args:
        console.print("[yellow]Usage: /ask <prompt>[/yellow]")
        return
    
    user_prompt = " ".join(args)
    prompt = generate_prompt("ask", user_prompt)
    if prompt:
        pyperclip.copy(prompt)
        console.print("[green]✓ Prompt copied to clipboard[/green]")

def command_test(args: List[str]) -> None:
    """Test command."""
    user_prompt = " ".join(args).strip() or None
    prompt = generate_prompt("test", user_prompt)
    if prompt:
        pyperclip.copy(prompt)
        console.print("[green]✓ Prompt copied to clipboard[/green]")

def review_code(code: str, language="groovy"):
    if not code.endswith("\n"):
        code += "\n"
    syntax = Syntax(
        code,
        language,
        theme="monokai",
        line_numbers=True,
        word_wrap=False
    )

    with console.pager():
        console.print(syntax)

# -------------------------
# Main TUI
# -------------------------

def main_tui():
    """Main TUI loop."""
    console.print(Panel.fit(
        "[bold cyan]Grails Prompt Compiler [/bold cyan]\n"
        "[dim]Created By LaPr[/dim]",
        border_style="cyan"
    ))
    
    if not JPYPE_AVAILABLE:
        console.print("[yellow]JPype not installed. Using regex parser only.[/yellow]")
        console.print("[dim]For best results: pip install jpype1[/dim]")
    
    console.print("[dim]Type /help for commands, /quit to exit[/dim]\n")
    
    while True:
        try:
            user_input = pt_prompt(
                "> ",
                completer=WordCompleter([
                    '/add', '/drop', '/method', '/refactor', '/ask',  
                    '/implement', '/test', '/status', '/help', '/quit'
                ], ignore_case=True, WORD= True),
                complete_style=CompleteStyle.MULTI_COLUMN
            ).strip()
            
            if not user_input:
                continue
            
            parts = user_input.split()
            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            if command == '/add':
                command_add(args)
            elif command == '/drop':
                command_drop()
            elif command == '/method':
                command_method(args)
            elif command == '/refactor':
                command_refactor(args)
            elif command == '/implement':
                command_implement(args)
            elif command == '/test':
                command_test(args)
            elif command == '/ask':
                command_ask(args)
            elif command == '/status':
                if CONTEXT_FILES:
                    console.print(f"[cyan]Files: {len(CONTEXT_FILES)}[/cyan]")
                    for file in CONTEXT_FILES:
                        console.print(f"[cyan]Files: {file}[/cyan]")
                    total_methods = sum(len(v) for v in METHOD_INDEX.values())
                    console.print(f"[cyan]Methods: {total_methods}[/cyan]")
                    if SELECTED_METHOD:
                        console.print(f"[green]Selected: {SELECTED_METHOD}[/green]")
                else:
                    console.print("[yellow]No context loaded[/yellow]")
            elif command == '/help':
                console.print(Panel("""
[bold]Commands:[/bold]
  /add <pattern>    - Add files (fuzzy find)
  /drop            - Clear all files
  /method [name]   - Select method
  /refactor <prompt> - Refactor selected method
  /implement <prompt> - Implement functionality
  /test <prompt>   - Generate Spock tests
  /status          - Show current context
  /quit            - Exit application

[dim]Examples:[/dim]
  /add service      # Find service files
  /method saveUser  # Select saveUser method
  /refactor "Extract validation logic"
                """, title="Help", border_style="blue"))
            elif command == '/quit':
                console.print("[green]Goodbye![/green]")
                if JVM_STARTED:
                    jpype.shutdownJVM()
                break
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except EOFError:
            console.print("\n[green]Goodbye![/green]")
            if JVM_STARTED:
                jpype.shutdownJVM()
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def install_dependencies():
    """Install required dependencies."""
    console.print("[cyan]Installing dependencies...[/cyan]")
    
    deps = [
        "jpype1",  # Note: package name is jpype1, import is jpype
        "rich",
        "pyperclip",
        "fuzzywuzzy",
        "python-Levenshtein",
        "prompt-toolkit"
    ]
    
    subprocess.run([sys.executable, "-m", "pip", "install"] + deps)
    
    console.print("\n[green]Installation complete![/green]")
    console.print("[yellow]You may also need to download groovy-all.jar[/yellow]")
    console.print("[dim]The app will try to find it automatically.[/dim]")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_dependencies()
    else:
        main_tui()