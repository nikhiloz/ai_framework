import ctypes
import os
import sys

class TransformerModelPy:
    def __init__(self, lib_path, mode="create", vocab_size=100, embed_dim=32, num_heads=4, num_blocks=2, ffn_dim=64, model_path=None):
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib.create_model_bridge.restype = ctypes.c_void_p
        self.lib.create_model_bridge.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        
        self.lib.load_gguf_model_bridge.restype = ctypes.c_void_p
        self.lib.load_gguf_model_bridge.argtypes = [ctypes.c_char_p]

        self.lib.free_model_bridge.argtypes = [ctypes.c_void_p]
        
        self.lib.generate_single_token_bridge.restype = ctypes.c_int
        self.lib.generate_single_token_bridge.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p
        ]

        self.lib.decode_token_bridge.restype = ctypes.c_char_p
        self.lib.decode_token_bridge.argtypes = [
            ctypes.c_void_p, ctypes.c_int
        ]
        
        if mode == "create":
            self.handle = self.lib.create_model_bridge(
                vocab_size, embed_dim, num_heads, num_blocks, ffn_dim
            )
        elif mode == "gguf":
            if not model_path:
                raise ValueError("model_path is required for gguf mode")
            self.handle = self.lib.load_gguf_model_bridge(model_path.encode('utf-8'))
            if not self.handle:
                raise RuntimeError(f"Failed to load GGUF model from {model_path}")
        
        if not self.handle:
            raise RuntimeError("Failed to create model handle")
            
        print(f"Model loaded via {mode} mode from {model_path if mode == 'gguf' else 'manual creation'}")

    def generate_token(self, prompt):
        prompt_bytes = prompt.encode('utf-8')
        return self.lib.generate_single_token_bridge(self.handle, prompt_bytes)

    def decode_token(self, token_id):
        decoded = self.lib.decode_token_bridge(self.handle, token_id)
        if decoded is None:
            return "NULL"
        return decoded.decode('utf-8')

    def __del__(self):
        if hasattr(self, 'handle'):
            self.lib.free_model_bridge(self.handle)

if __name__ == "__main__":
    # Use absolute path to the library
    lib_path = "/data/data/com.termux/files/home/ai_framework/libaiframework.so"
    
    if not os.path.exists(lib_path):
        print(f"Error: Library not found at {lib_path}")
        sys.exit(1)

    print("--- Testing Manual Model Creation ---")
    try:
        model = TransformerModelPy(lib_path, mode="create")
        token = model.generate_token("hello world")
        print(f"Generated token ID: {token}")
        try:
            decoded = model.decode_token(token)
            print(f"Decoded token: {decoded}")
        except Exception as e:
            print(f"Decoding failed: {e}")
    except Exception as e:
        print(f"Manual creation test failed: {e}")

    print("\n--- Testing GGUF Model Loading ---")
    gguf_path = "/data/data/com.termux/files/home/ai_framework/tinyllama.gguf"
    if os.path.exists(gguf_path):
        try:
            model_gguf = TransformerModelPy(lib_path, mode="gguf", model_path=gguf_path)
            token_gguf = model_gguf.generate_token("Hello")
            print(f"Generated token ID: {token_gguf}")
            decoded_gguf = model_gguf.decode_token(token_gguf)
            print(f"Decoded token: {decoded_gguf}")
        except Exception as e:
            print(f"GGUF loading test failed: {e}")
    else:
        print(f"Skipping GGUF test: {gguf_path} not found")
