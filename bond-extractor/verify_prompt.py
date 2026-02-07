import inspect
import app

def verify_prompt():
    source = inspect.getsource(app.call_mistral_legai)
    
    expected_snippet_start = "SYSTEM ROLE"
    expected_snippet_middle = "STRICT DATA SOURCE RULES (MANDATORY)"
    expected_snippet_end = "Application-controlled knowledge boundaries"
    
    if expected_snippet_start in source and expected_snippet_middle in source and expected_snippet_end in source:
        print("SUCCESS: New system prompt found in call_mistral_legai code.")
        
        # Extract the prompt variable roughly
        start_idx = source.find('system_prompt = """')
        if start_idx == -1:
             print("WARNING: Could not find 'system_prompt = \"\"\"' assignment.")
        else:
             print("Found system_prompt assignment.")
    else:
        print("FAILURE: New system prompt NOT found in call_mistral_legai code.")
        print("Source excerpt:")
        print(source[:500] + "..." + source[-500:])

if __name__ == "__main__":
    verify_prompt()
