import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .parser import SimpleParser

class SimpleReplacer:
    def __init__(self):
        self.parser = SimpleParser()
        pass
    
    def replace_method_in_file(self, file_path: Path, method_name: str, new_method_body: str, 
                              class_name: Optional[str] = None) -> bool:
        """
        Replace a method in a Groovy file with new content.
        
        Args:
            file_path: Path to the Groovy file
            method_name: Name of the method to replace
            new_method_body: Complete new method code (including signature and body)
            class_name: Optional class name if method might be in nested classes
            
        Returns:
            True if replacement was successful, False otherwise
        """
        try:
            # Read the original file
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parse methods to find the target
            methods = self.parser.parse_with_state_machine(source, file_path)
            
            # Find the method to replace
            target_method = None
            for method in methods:
                if method["method"] == method_name:
                    # If class_name is specified, check it matches
                    if class_name and method["class"] != class_name:
                        continue
                    target_method = method
                    break
            
            if not target_method:
                print(f"Method '{method_name}' not found in file {file_path}")
                return False
            
            # Validate that new_method_body starts with the same signature
            # Extract signature from the new method
            new_signature = self.extract_method_signature(new_method_body)
            if not new_signature:
                print(f"Could not extract signature from new method body")
                return False
            
            # Get the lines of the original file
            lines = source.split('\n')
            
            # Build the new file content
            new_lines = []
            
            # Add lines before the method
            new_lines.extend(lines[:target_method["start_line"]])
            
            # Add the new method
            # Split the new method body into lines to preserve formatting
            new_method_lines = new_method_body.split('\n')
            new_lines.extend(new_method_lines)
            
            # Add lines after the method
            if target_method["end_line"] + 1 < len(lines):
                new_lines.extend(lines[target_method["end_line"] + 1:])
            
            # Write back to file
            new_source = '\n'.join(new_lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_source)
            
            print(f"Successfully replaced method '{method_name}' in {file_path}")
            return True
            
        except Exception as e:
            print(f"Error replacing method: {e}")
            return False
    
    def extract_method_signature(self, method_body: str) -> Optional[str]:
        """Extract just the method signature from a method body."""
        lines = method_body.strip().split('\n')
        if not lines:
            return None
        
        # Find the first line that contains '(' and ends with ')' or has opening brace
        signature_lines = []
        found_brace = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            signature_lines.append(line)
            
            # Check if we've found the complete signature
            # Signature is complete when we have '(' and ')' or '{'
            if '(' in stripped:
                # Check if we have closing parenthesis or opening brace
                if ')' in stripped or '{' in stripped:
                    break
                # For multiline signatures, continue until we find ')' or '{'
                continue
            
            # If we find opening brace without parentheses, this might be an invalid signature
            if '{' in stripped:
                if '(' not in signature_lines[0]:
                    return None  # Not a valid method signature
                break
        
        return '\n'.join(signature_lines)
    

    def replace_method_with_llm_response(self, file_path: Path, method_name: str, 
                                        llm_response: str) -> bool:
        """
        Parse LLM response and replace method.
        
        Expected llm_response format (JSON):
        {
            "refactored_method": "full method code here",
            "changes_made": ["list of changes"],
            "ast_anchors": {"key_method_elements": "locations"}
        }
        """
        import json
        breakpoint()
        
        try:
            # Parse the JSON response
            response_data = json.loads(llm_response)
            breakpoint()
            
            # Extract the refactored method
            if "refactored_method" not in response_data:
                print("No 'refactored_method' found in LLM response")
                return False
            
            refactored_method = response_data["refactored_method"]
            
            # Optional: Log the changes made
            if "changes_made" in response_data:
                print(f"Changes to be applied: {response_data['changes_made']}")
            
            # Replace the method
            return self.replace_method_in_file(file_path, method_name, refactored_method)
            
        except json.JSONDecodeError:
            # If it's not JSON, assume it's just the method code
            print("LLM response is not JSON, trying as raw method code")
            return self.replace_method_in_file(file_path, method_name, llm_response)

# Usage example:
if __name__ == "__main__":
    replacer = SimpleReplacer()
    breakpoint()
    
    # Example usage with LLM response
    llm_response_json = """{
        "changes_made": ["Added null guard clause at method entry"]
    }"""

    import json

    llm_response_json = re.sub(
        r'"refactored_method": "(.*?)",',
        lambda m: f'"refactored_method": {json.dumps(m.group(1))},',
        llm_response_json,
        flags=re.DOTALL
    )
    
    # Replace the method
    success = replacer.replace_method_with_llm_response(
        file_path=Path(r"E:\workspace\assist\test/UserService.groovy"),
        method_name="createUpdateOfficerAddress",
        llm_response=llm_response_json
    )
    
    if success:
        print("Method replacement successful!")