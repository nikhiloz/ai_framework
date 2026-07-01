#!/bin/bash
cd /data/data/com.termux/files/home/ai_framework
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/data/com.termux/files/home/llama.cpp/build/bin
python3 -u ai_cli.py
