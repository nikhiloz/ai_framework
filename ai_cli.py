import ctypes
import sys
import os

# Configuration
LIB_PATH = "/data/data/com.termux/files/home/ai_framework/libaiframework.so"

class AIModel:
    def __init__(self):
        if not os.path.exists(LIB_PATH):
            print(f"Error: Library not found at {LIB_PATH}")
            sys.exit(1)
        
        self.lib = ctypes.CDLL(LIB_PATH)
        
        # Define function signatures
        self.lib.create_model_bridge.restype = ctypes.c_void_p
        self.lib.create_model_bridge.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        
        self.lib.free_model_bridge.argtypes = [ctypes.c_void_p]
        
        self.lib.generate_single_token_bridge.restype = ctypes.c_int
        self.lib.generate_single_token_bridge.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p
        ]
        
        self.lib.decode_token_bridge.restype = ctypes.c_char_p
        self.lib.decode_token_bridge.argtypes = [
            ctypes.c_void_p, ctypes.c_int
        ]
        
        # Initialize a small model for the CLI
        self.handle = self.lib.create_model_bridge(100, 32, 4, 2, 64)

    def generate_stream(self, prompt, max_tokens=100):
        current_prompt = prompt
        generated_text = ""
        
        for _ in range(max_tokens):
            token_id = self.lib.generate_single_token_bridge(self.handle, current_prompt.encode('utf-8'))
            if token_id == 0: # Assume 0 is EOS
                break
                
            char = self.lib.decode_token_bridge(self.handle, token_id).decode('utf-8')
            print(char, end='', flush=True)
            
            generated_text += char
            current_prompt += char
            
        print()
        return generated_text

    def __del__(self):
        if hasattr(self, 'handle'):
            self.lib.free_model_bridge(self.handle)

def main():
    model = AIModel()
    print("--- Local AI CLI (Ollama-style) ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            user_input = input("
>>> ")
            if user_input.lower() in ('exit', 'quit'):
                break
            if not user_input.strip():
                continue
            
            print("AI: ", end='')
            model.generate_stream(user_input)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
