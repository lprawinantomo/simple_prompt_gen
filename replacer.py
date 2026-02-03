import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .parser import SimpleParser

class SimpleReplacer:
    def __init__(self):
        self.parser = SimpleParser()
        pass
    
    def replace_method_in_file(self, selected_method_info: Dict, new_method_body: str) -> bool:
        """
        Replace a method in a Groovy file with new content.
        
        Args:
            selected_method_info: Dictionary containing method information including file, start_line, end_line
            new_method_body: Complete new method code (including signature and body)
            
        Returns:
            True if replacement was successful, False otherwise
        """
        try:
            # Read the original file
            with open(selected_method_info["file"], 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Get the lines of the original file
            lines = source.splitlines(keepends=False)
            
            # Get method boundaries
            start_line = selected_method_info["start_line"]
            end_line = selected_method_info["end_line"]
            
            # Build the new file content
            new_lines = []
            
            # Add lines before the method
            if start_line > 0:
                new_lines.extend(lines[:start_line])

            new_method_body = self.indent_if_needed(new_method_body)            

            # Add the new method with proper indentation
            new_method_lines = new_method_body.splitlines(keepends=False)
            
            # Clean up and indent the new method body
            # Remove leading/trailing empty lines
            new_method_lines = [line for line in new_method_lines if line.strip() != '']
            # Add tab to each line
            #indented_method_lines = ['    ' + line for line in new_method_lines]
            new_lines.extend(new_method_lines)
            
            # Add lines after the method
            if end_line + 1 < len(lines):
                new_lines.extend(lines[end_line + 1:])
            
            # Write back to file
            new_source = '\n'.join(new_lines)
            with open(selected_method_info["file"], 'w', encoding='utf-8') as f:
                f.write(new_source)
            
            print(f"Successfully replaced method '{selected_method_info['method']}' in {selected_method_info['file']}")
            return True
            
        except Exception as e:
            print(f"Error replacing method: {e}")
            return False


    def indent_if_needed(self, s) -> str:
        if not s.startswith((' ', '\t')):
            return '    ' + s
        return s

    
    def replace_method_with_llm_response(self, selected_method_info: Dict, 
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
            return self.replace_method_in_file(selected_method_info, refactored_method)
            
        except json.JSONDecodeError:
            # If it's not JSON, assume it's just the method code
            print("LLM response is not JSON, trying as raw method code")
            return self.replace_method_in_file(selected_method_info, llm_response)

# Usage example:
if __name__ == "__main__":
    replacer = SimpleReplacer()
    breakpoint()
    
    # Example usage with LLM response
    llm_response_json = """{
        "changes_made": ["Added null guard clause at method entry"]
    }"""