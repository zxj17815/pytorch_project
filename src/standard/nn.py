import os
import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from tqdm import tqdm

from data_londer import IrisDataLoader


class IrisModel(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        super(IrisModel, self).__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# 定义计算环境
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  # mac环境
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 定义数据集
custom_dataset = IrisDataLoader(csv_path=os.path.join(
    os.path.dirname(__file__), "data/iris.csv"))

# 训练集
train_size = int(len(custom_dataset) * 0.7)
# 验证集
val_size = int(len(custom_dataset) * 0.2)
# 测试集
test_size = len(custom_dataset) - train_size - val_size

# 对数据集进行随机划分
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    custom_dataset, [train_size, val_size, test_size])

# 定义数据加载器
train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True)  # shuffle 是否对剩余的数据进行洗牌
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

print("训练集大小:", len(train_dataloader)*16)
print("验证集大小:", len(val_dataloader))
print("测试集大小:", len(test_dataloader))


def inference(model, dataset, device):  # 定义推理函数 计算准确率
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            acc_num += (predicted == labels).sum().item()
    return acc_num / len(dataset)


def main(lr=0.005, epochs=20):
    model = IrisModel(in_dim=4, hidden_dim1=12, hidden_dim2=6, out_dim=3)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]  # 判断可以迭代的 需要训练的参数
    optimizer = optim.Adam(pg, lr=lr)  # 定义优化器

    # 权重文件
    save_path = os.path.join(os.path.dirname(__file__), "model/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0

        train_bar = tqdm(train_dataloader, file=sys.stdout, ncols=100)
        for datas in train_bar:
            data, label = datas
            label = label.squeeze(-1)
            sample_num += data.shape[0]

            optimizer.zero_grad()
            outputs = model(data.to(device))
            pred_class = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(pred_class, label.to(device)).sum()

            loss = loss_fn(outputs, label.to(device))
            loss.backward()
            optimizer.step()

            train_acc = acc_num / sample_num

            train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(
                epoch + 1,
                epochs,
                loss
            )
        # 计算验证集准确率
        val_acc = inference(model, val_dataloader, device)
        print("train epoch[{}/{}] loss:{:.4f} train_acc:{:.4f} val_acc:{:.4f}".format(
            epoch + 1, epochs, loss, train_acc.item(), val_acc))

        torch.save(model.state_dict(), os.path.join(save_path, "nn.pth"))

        # 每次迭代后对初始化数据清空
        train_acc = 0
        val_acc = 0
    print("训练完成")

    test_acc = inference(model, test_dataloader, device)
    print("测试集准确率:", test_acc)


if __name__ == "__main__":
    main()
