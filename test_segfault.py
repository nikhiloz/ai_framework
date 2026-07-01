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
        
        self.lib.load_gguf_model_bridge.restype = ctypes.c_void_p
        self.lib.load_gguf_model_bridge.argtypes = [ctypes.c_char_p]
        self.lib.free_model_bridge.argtypes = [ctypes.c_void_p]
        self.lib.generate_single_token_bridge.restype = ctypes.c_int
        self.lib.generate_single_token_bridge.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.decode_token_bridge.restype = ctypes.c_char_p
        self.lib.decode_token_bridge.argtypes = [ctypes.c_void_p, ctypes.c_int]
        
        print(f"Loading model from {model_path}...")
        self.handle = self.lib.load_gguf_model_bridge(model_path.encode('utf-8'))
        if not self.handle:
            print("Error: Failed to load GGUF model.")
            sys.exit(1)
        print("Model loaded successfully!")

    def generate(self, prompt, max_tokens=10):
        current_prompt = prompt
        generated_text = ""
        print(f"Generating for prompt: {prompt}")
        for i in range(max_tokens):
            token_id = self.lib.generate_single_token_bridge(self.handle, current_prompt.encode('utf-8'))
            if token_id == 0:
                break
            char = self.lib.decode_token_bridge(self.handle, token_id).decode('utf-8')
            print(f"Token {i}: {char}")
            generated_text += char
            current_prompt += char
        return generated_text

    def __del__(self):
        if hasattr(self, 'handle'):
            self.lib.free_model_bridge(self.handle)

if __name__ == "__main__":
    model = AIModel()
    result = model.generate("hi")
    print(f"Result: {result}")
