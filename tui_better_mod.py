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
from .parser import SimpleParser

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
# Initialize Parser
# -------------------------

PARSER = SimpleParser()

# -------------------------
# TUI Commands (Same as before)
# -------------------------

def fuzzy_find_files(pattern: str) -> List[Path]:
    """Fuzzy find Groovy files (cross-platform)."""
    if not pattern.strip():
        return []

    root_path = Path(
        os.getenv("PROJECT_ROOT", Path.cwd())
    ).expanduser().resolve()

    all_files = list(root_path.rglob("*.groovy"))

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
    
    project_root = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()

    table = Table(title="Matching Files")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("File Path", style="green")
    table.add_column("#", style="cyan", justify="right")

    for i, file in enumerate(matched_files, 1):
        file_path = Path(file).resolve()
        relative_path = file_path.relative_to(project_root)
        table.add_row(str(i), str(relative_path), str(i))

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
        METHOD_INDEX, CLASS_METHODS = PARSER.index_files(CONTEXT_FILES)

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
        METHOD_INDEX, CLASS_METHODS = PARSER.index_files(CONTEXT_FILES)
    
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
        graph = PARSER.expand_call_graph(SELECTED_METHOD_INFO, METHOD_INDEX, CLASS_METHODS, call_depth)
        
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