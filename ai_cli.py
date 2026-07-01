import ctypes
import sys
import os

# Configuration
LIB_PATH = "/data/data/com.termux/files/home/ai_framework/libaiframework.so"

class AIModel:
    def __init__(self, model_path="tinyllama.gguf"):
        if not os.path.exists(LIB_PATH):
            print(f"Error: Library not found at {LIB_PATH}")
            sys.exit(1)
        
        self.lib = ctypes.CDLL(LIB_PATH)
        
        # Define function signatures
        self.lib.load_gguf_model_bridge.restype = ctypes.c_void_p
        self.lib.load_gguf_model_bridge.argtypes = [ctypes.c_char_p]
        
        self.lib.free_model_bridge.argtypes = [ctypes.c_void_p]
        
        # New bridge functions for incremental generation
        self.lib.init_generation_bridge.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.generate_next_token_bridge.restype = ctypes.c_int
        self.lib.generate_next_token_bridge.argtypes = [ctypes.c_void_p]
        
        self.lib.decode_token_bridge.restype = ctypes.c_char_p
        self.lib.decode_token_bridge.argtypes = [
            ctypes.c_void_p, ctypes.c_int
        ]
        
        # Load the real GGUF model
        print(f"DEBUG: Loading model from {model_path}...")
        sys.stdout.flush()
        self.handle = self.lib.load_gguf_model_bridge(model_path.encode('utf-8'))
        if not self.handle:
            print("Error: Failed to load GGUF model.")
            sys.exit(1)
        print("Model loaded successfully!")

    def generate_stream(self, prompt, max_tokens=100):
        # Initialize generation with prompt
        self.lib.init_generation_bridge(self.handle, prompt.encode('utf-8'))
        
        generated_text = ""
        
        for _ in range(max_tokens):
            # Generate just the next token
            token_id = self.lib.generate_next_token_bridge(self.handle)
            if token_id <= 2: # Assume EOS or special tokens <= 2
                break
                
            char = self.lib.decode_token_bridge(self.handle, token_id).decode('utf-8')
            print(char, end='', flush=True)
            
            generated_text += char
            
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
            user_input = input("\n>>> ")
            if user_input.lower() in ('exit', 'quit'):
                break
            if not user_input.strip():
                continue
            
            print("AI: ", end='')
            model.generate_stream(user_input)
            
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
