import ctypes
import os

class TransformerModelPy:
    def __init__(self, lib_path, vocab_size=100, embed_dim=32, num_heads=4, num_blocks=2, ffn_dim=64):
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib.create_model_bridge.restype = ctypes.c_void_p
        self.lib.create_model_bridge.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        
        self.lib.free_model_bridge.argtypes = [ctypes.c_void_p]
        
        self.lib.generate_token_bridge.restype = ctypes.c_int
        self.lib.generate_token_bridge.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
        ]
        
        self.handle = self.lib.create_model_bridge(
            vocab_size, embed_dim, num_heads, num_blocks, ffn_dim
        )
        print(f"Model loaded from {lib_path}")

    def generate_token(self, prompt, max_len=10):
        prompt_bytes = prompt.encode('utf-8')
        return self.lib.generate_token_bridge(self.handle, prompt_bytes, max_len)

    def __del__(self):
        if hasattr(self, 'handle'):
            self.lib.free_model_bridge(self.handle)

if __name__ == "__main__":
    # Test the bridge
    lib_path = "./ai_framework/libaiframework.so"
    if os.path.exists(lib_path):
        model = TransformerModelPy(lib_path)
        token = model.generate_token("hello world")
        print(f"Generated token ID: {token}")
    else:
        print(f"Library not found at {lib_path}")
