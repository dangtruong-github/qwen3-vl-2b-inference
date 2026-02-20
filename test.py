import numpy as np

txt_path_to_read = "/home/dangtruongdefault/qwen3-vl-2b-inference/scripts/run_out.txt"

# Use a list comprehension to feed NumPy
with open(txt_path_to_read, 'r') as f:
    data = [np.array(line.split(), dtype=float) for line in f]

# Accessing the second number of the first row
c16 = data[2]
c8 = data[5]

print(c16 - c8)
    