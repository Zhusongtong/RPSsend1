# A股基于欧奈尔 RPS 量化选股

## 1. 运行 `prps--all.py` 文件
运行 `prps--all.py` 文件以启动量化选股程序。

---

## 2. 运行设置
在运行程序之前，可以根据需求进行以下设置：

### 市场选择
- 默认 `market = 0`：
  - `0`：全部市场
  - `1`：创业板（CYB）
  - `2`：中证 500（ZZ500）
  - `3`：国证 2000（GZ2000）
  - `4`：沪深 300（HS300）
  - `5`：中证 1000（ZZ1000）

### 天数
- 默认 `day = 130`：计算 RPS 的天数为 130 天。

### 日期
- 默认 `today = time.strftime("%Y-%m-%d")`：使用当天时间。
- 可以手动设置回测时间，例如：`today = '2024-12-17'`。

---

## 3. 输出结果
程序运行后，会根据 RPS 筛选结果输出选股列表。

### 示例图片
![RPS 筛选结果]([https://github.com/user-attachments/assets/eb638cf7-540f-408f-9061-4c9ad7b28bc3/600x400)  
---

## 4. 对接飞书机器人
将筛选结果通过飞书机器人发送。
<img width="122" alt="1735281932549" src="https://github.com/user-attachments/assets/1eeccc88-9b31-41d9-80a2-274d02940ade" />
---

## 使用步骤
1. 运行 `prps--all.py` 文件。
2. 根据需求设置市场、天数和日期。
3. 查看 RPS 筛选结果。
4. 将结果对接飞书机器人。

---

## 注意事项
- 确保 Python 环境已安装所需依赖。
- 日期格式必须为 `YYYY-MM-DD`。
- 飞书机器人需要提前配置并获取 ID。

---

## 联系方式
如有问题，请联系：`https://nyseai.cn`
扫描下方二维码，关注我们的微信公众号，获取更多量化投资资讯和工具：

<img width="129" alt="image" src="https://github.com/user-attachments/assets/b4cadf73-a7ca-4116-8614-337d9d7b07d4" />


---
