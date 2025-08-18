import pandas as pd
import re
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python analyze_success.py <input_txt> [output_csv] [output_png]")
    sys.exit(1)

input_txt = sys.argv[1]
output_csv = sys.argv[2] if len(sys.argv) > 2 else "success_summary.csv"
output_png = sys.argv[3] if len(sys.argv) > 3 else None

# 更宽泛的 task 捕捉方式，兼容中划线
pattern = r"steps_(\d+)_pytorch_model_infer_(.+)-v0\.log\.run(\d+) → Average success: ([0-9.]+)"

data = []
with open(input_txt, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            step, task, run_id, score = match.groups()
            data.append({
                'step': int(step),
                'task': task,
                'run_id': int(run_id),
                'score': float(score)
            })

df = pd.DataFrame(data)

if df.empty:
    print("❌ No valid data found in input file.")
    sys.exit(1)

# 聚合：对同一 step、task 的多 run 求平均
avg_df = (
    df.groupby(['step', 'task'])['score']
    .mean()
    .unstack()
    .sort_index()
)

# 添加总平均列
avg_df['Average Across Tasks'] = avg_df.mean(axis=1)

# 保存 CSV
avg_df.to_csv(output_csv)
print(f"✅ CSV saved to {output_csv}")

# 可视化
if output_png:
    plt.figure(figsize=(10, 6))
    for column in avg_df.columns:
        if column == 'Average Across Tasks':
            plt.plot(avg_df.index, avg_df[column], label=column, color='red', linewidth=2)
        else:
            plt.plot(
                avg_df.index,
                avg_df[column],
                linestyle='--',
                alpha=0.5,
                label=column
            )
    plt.xlabel("Training Step")
    plt.ylabel("Average Success")
    plt.title("Success Rate by Task and Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"📈 Plot saved to {output_png}")
