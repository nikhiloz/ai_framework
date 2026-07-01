import struct
import sys

def read_gguf(filepath):
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Header: Magic=0x{magic:X}, Version={version}, Tensors={tensor_count}, KVs={kv_count}")
        
        for i in range(kv_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            val_type = struct.unpack('<I', f.read(4))[0]
            print(f"KV {i}: Key='{key}', Type={val_type}")
            
            # Skip value
            if val_type == 0: f.read(1) # uint8
            elif val_type == 1: f.read(1) # int8
            elif val_type == 2: f.read(2) # uint16
            elif val_type == 3: f.read(2) # int16
            elif val_type == 4: f.read(4) # uint32
            elif val_type == 5: f.read(4) # int32
            elif val_type == 6: f.read(4) # float32
            elif val_type == 7: f.read(1) # bool
            elif val_type == 8: # string
                s_len = struct.unpack('<Q', f.read(8))[0]
                f.read(s_len)
            elif val_type == 9: # array
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                print(f"  -> Array: {arr_len} elements of type {arr_type}")
                # This is the hard part, need to know element size
                # For now just skip what we can or stop
                break 
            elif val_type == 10: f.read(8) # uint64
            elif val_type == 11: f.read(8) # int64
            elif val_type == 12: f.read(8) # float64

read_gguf("tinyllama.gguf")
