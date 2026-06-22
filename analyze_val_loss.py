import re

# 读取日志文件并提取每个epoch的验证loss
with open("train.log", 'r') as f:
    lines = f.readlines()

# 存储每个epoch的验证loss
epoch_losses = {}

# 匹配模式: "Epoch X, Validate shapenet, Loss: Y"
pattern = r'Epoch (\d+), Validate shapenet, Loss: ([\d.]+)'

for line in lines:
    match = re.search(pattern, line)
    if match:
        epoch = int(match.group(1))
        loss = float(match.group(2))
        if epoch not in epoch_losses:
            epoch_losses[epoch] = []
        epoch_losses[epoch].append(loss)

# 计算每个epoch的平均loss
print("Epoch\tAverage Val Loss\tMin Loss\tMax Loss\tSamples")
print("-" * 70)
for epoch in sorted(epoch_losses.keys())[:100]:  # 显示前100个epoch
    losses = epoch_losses[epoch]
    avg_loss = sum(losses) / len(losses)
    min_loss = min(losses)
    max_loss = max(losses)
    print(f"{epoch}\t{avg_loss:.6f}\t{min_loss:.6f}\t{max_loss:.6f}\t{len(losses)}")

# 趋势分析
print("\n=== 趋势分析 ===")
epochs = sorted(epoch_losses.keys())
avg_losses = [sum(epoch_losses[e])/len(epoch_losses[e]) for e in epochs]

# 找到最小loss的epoch
min_idx = avg_losses.index(min(avg_losses))
print(f"最小验证loss: Epoch {epochs[min_idx]}, Loss = {avg_losses[min_idx]:.6f}")

# 检查是否过拟合（最后几个epoch比前面的高）
if len(epochs) >= 5:
    recent_avg = sum(avg_losses[-5:]) / 5
    early_avg = sum(avg_losses[:5]) / 5
    print(f"前5个epoch平均loss: {early_avg:.6f}")
    print(f"最后5个epoch平均loss: {recent_avg:.6f}")
    if recent_avg > early_avg:
        print("趋势: 验证loss在后期上升，可能存在过拟合")
    else:
        print("趋势: 验证loss持续下降")
