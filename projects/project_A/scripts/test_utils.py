import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# サンプルデータフレームの作成
data = {'Category': ['A', 'B', 'C', 'D', 'E'],
        'Value': np.random.randint(1, 100, 5)}
df = pd.DataFrame(data)

print("サンプルデータフレーム:")
print(df)

# データの可視化 (棒グラフ)
plt.figure(figsize=(8, 5))
plt.bar(df['Category'], df['Value'], color=['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral'])
plt.title('Sample Bar Chart')
plt.xlabel('Category')
plt.ylabel('Value')
plt.grid(axis='y', linestyle='--')
plt.show()

print("\nサンプル分析が完了しました。グラフが表示されていることを確認してください。") 