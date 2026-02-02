# -------------------------
# Groovy AST Parser using JPype
# -------------------------
import sys
import os
import re
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Dict, Set, Optional, Tuple, Any
from rich.console import Console
# Initialize Rich console
console = Console()

# -------------------------
# Indexing and Call Graph
# -------------------------

class SimpleParser:
    """Parse Groovy files."""
    
    def __init__(self):
        self.jvm_started = False
        self.groovy_jar = None

    def index_files(self, files: List[Path]) -> Tuple[Dict, Dict]:
        """Index Groovy files."""
        console.print(f"[cyan]Indexing {len(files)} files...[/cyan]")
        
        method_index = defaultdict(list)
        class_methods = defaultdict(dict)
        total_methods = 0
        
        for file in files:
            methods = self.parse_file(file)
            
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
    
    def expand_call_graph(self, entry_method: Dict, method_index: Dict, 
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
            calls = self.extract_method_calls_with_context(
                current.get("source", ""),
                current_class,
                class_methods
            )
            
            symbols = self.build_symbol_table(current)
            
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
                    resolved = self.resolve_receiver_call(receiver, method, symbols, class_methods)
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

    def resolve_receiver_call(self, receiver: str, method: str, symbols: Dict, class_methods: Dict) -> Optional[Dict]:
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

    def extract_method_calls_with_context(self, source: str, current_class: str, 
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

    def build_symbol_table(self, method_info: Dict) -> Dict[str, str]:
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