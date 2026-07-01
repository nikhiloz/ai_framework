import subprocess
p = subprocess.Popen(['./run_ai.sh'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="ai_framework")
stdout, stderr = p.communicate(input='hi\n')
print(stdout)
