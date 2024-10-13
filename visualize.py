import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'out.txt'
with open(file_path, 'r') as file:
    file_contents = file.readlines()

data = []
for line in file_contents:
    match = re.search(r'\|\s+(\d+\.\d+)\s+TFLOPS\s+@\s+(\d+)x(\d+)x(\d+)', line)
    if match:
        tflops = float(match.group(1))
        m = int(match.group(2))
        n = int(match.group(3))
        k = int(match.group(4))
        data.append([tflops, m, n, k])

df = pd.DataFrame(data, columns=['TFLOPS', 'M', 'N', 'K'])

heatmap_data = pd.pivot_table(df, values='TFLOPS', index='M', columns='N', aggfunc=np.mean)

plt.figure(figsize=(12, 12))

plt.imshow(heatmap_data, cmap='rainbow', aspect='auto', origin='lower')

plt.colorbar(label='TFLOPS')
plt.title('TFLOPS Heatmap (M vs N)')
plt.xlabel('N (Dimension)')
plt.ylabel('M (Dimension)')

plt.show()

top_10_flops = df.sort_values(by='TFLOPS', ascending=False).head(20)

print(top_10_flops)
