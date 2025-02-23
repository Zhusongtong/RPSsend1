import pandas as pd
import requests
from datetime import datetime

# 读取 CSV 文件
file_path = "1013.csv"
df = pd.read_csv(file_path)

# 确保日期列解析正确
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 仅转换数值列，跳过日期列和 K 列
K_col_name = "name"  # 确保 K 列列名正确
for col in df.columns:
    if col not in ['date', K_col_name]:  # 保护 K 列不被误转换
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 确保 K 列数据不是 NaN，并转换为字符串
df[K_col_name] = df[K_col_name].astype(str).fillna("缺失值")

# 获取今天的日期
today = pd.Timestamp(datetime.now().date())

# 计算距离今天最近的日期
df['date_diff'] = (df['date'] - today).abs()
nearest_date_df = df[df['date_diff'] == df['date_diff'].min()]  # 筛选出最近日期

# 确定 Q 列索引（Q=17 假设）
Q_col_name = df.columns[16]  # 第 17 列

# 筛选符合 Q 列范围的数据
filtered_df = nearest_date_df[(nearest_date_df[Q_col_name] >= 75) & (nearest_date_df[Q_col_name] <= 93)]

# **按照 Q 列升序排列**
filtered_df = filtered_df.sort_values(by=Q_col_name, ascending=True)

# 输出符合条件的数据
output_lines = []
for index, row in filtered_df.iterrows():
    K_value = str(row[K_col_name])  # 确保 K 列转换为字符串
    Q_value = round(row[Q_col_name], 2)  # Q 列四舍五入
    output_lines.append(f"{K_value} rps120={Q_value}")

# 生成推送内容
output_text = "\n".join(output_lines)
print(output_text)  # 先在终端检查输出

# ============================
# **Server酱 推送到微信**
# ============================

sendkey = "SCT270757TGYSsmGdzCUzGJAgPDN8kaH1E"  # 你的 SendKey
server_url = f"https://sctapi.ftqq.com/{sendkey}.send"

# 推送标题
title = "股票筛选结果"
# 推送内容（以 Markdown 格式发送）
content = f"**今日符合筛选条件的股票**：\n\n```\n{output_text}\n```"

# 发送请求
data = {
    "title": title,
    "desp": content
}

response = requests.post(server_url, data=data)

# 检查推送是否成功
if response.status_code == 200:
    print("✅ 消息已成功推送到微信！")
else:
    print("❌ 消息推送失败，请检查 SendKey 是否正确！")
    print("错误信息:", response.text)
