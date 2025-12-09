import matplotlib.pyplot as plt
import numpy as np

# 數據輸入
metrics = [
    "Replacement Rate", 
    "Whitelist Coverage", 
    "Semantic Preservation", 
    "BERT Confidence", 
    "BERT F1"
]

# 10 句測試集數據 (將百分比轉換為 0-1 區間)
controlled_data = np.array([0.900, 0.467, 0.698, 0.890, 0.979]) 

# 100 句測試集數據 (將百分比轉換為 0-1 區間)
real_data = np.array([0.600, 0.111, 0.821, 0.802, 0.986])

x = np.arange(len(metrics))
width = 0.35 

fig, ax = plt.subplots(figsize=(12, 6))

# 繪製長條圖
rects1 = ax.bar(x - width/2, controlled_data, width, label='10 Sentences (Controlled)', color='#1f77b4')
rects2 = ax.bar(x + width/2, real_data, width, label='100 Sentences (Real)', color='#ff7f0e')

# 設定 Y 軸範圍和標籤
ax.set_ylim(0, 1.0)
ax.set_ylabel('Performance Score (0.0 to 1.0)', fontsize=12)
ax.set_title('Slangify 系統性能對比：受控測試與真實語料庫挑戰', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15, ha="right", fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 標註數據值
def autolabel(rects, is_percentage):
    for rect in rects:
        height = rect.get_height()
        if is_percentage:
            # 應用於 Rate 和 Coverage (前兩個指標)
            text = f'{height*100:.1f}%'
        else:
            # 應用於其他指標
            text = f'{height:.3f}'
            
        ax.annotate(text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

# Replacement Rate 和 Whitelist Coverage 是百分比
autolabel(rects1[0:2], True)
autolabel(rects2[0:2], True)

# Semantic Preservation, BERT Confidence, BERT F1 是小數
autolabel(rects1[2:], False)
autolabel(rects2[2:], False)


fig.tight_layout()

# 儲存圖片
plt.savefig('performance_comparison.png')
print("✅ 性能對比圖已成功保存為 performance_comparison.png")
