# 导入必要的库
import struct
import random
import socket
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from torchmetrics import Accuracy, Precision
from torchmetrics.classification import BinaryPrecision, BinaryAccuracy
attack_cat_values = ['Benign', 'Reconnaissance', 'DDoS', 'DoS', 'Theft']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F


def calculate_FAR(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    FAR = fp / (fp + tn)
    return FAR

def calculate_class_accuracies(true_labels, pred_labels):
    # 获取混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    # 计算每个类别的准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    return class_accuracies

class BotDataset(Dataset):
    def __init__(self, X_data, y_data):
        super().__init__()
        #X_data = X_data.astype(float)
        self.X_data_tensor = torch.from_numpy(X_data)
        self.y_data_tensor = torch.from_numpy(y_data)

    def __len__(self):
        return len(self.X_data_tensor)

    def __getitem__(self, index):
        return (self.X_data_tensor[index], self.y_data_tensor[index])

# 读取不同数据集

data_path = 'data/NF-BoT-IoT.csv'
df = pd.read_csv(data_path)
print(df['Label'].dtype)
df['IPV4_SRC_ADDR'] = df.IPV4_SRC_ADDR.apply(lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))
df['IPV4_SRC_ADDR'] = df.IPV4_SRC_ADDR.apply(str)
df['L4_SRC_PORT'] = df.L4_SRC_PORT.apply(str)
df['IPV4_DST_ADDR'] = df.IPV4_DST_ADDR.apply(str)
df['L4_DST_PORT'] = df.L4_DST_PORT.apply(str)
df['IPV4_SRC_ADDR'] = df['IPV4_SRC_ADDR'] + ':' + df['L4_SRC_PORT']
df['IPV4_DST_ADDR'] = df['IPV4_DST_ADDR'] + ':' + df['L4_DST_PORT']
df.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT'], inplace=True)
# 数据预处理
# 将分类变量转换为数值型
categorical_columns = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR','TCP_FLAGS', 'L7_PROTO', 'PROTOCOL']  # 请根据实际情况修改这个列表
for column in categorical_columns:
    df[column] = LabelEncoder().fit_transform(df[column])

X_cols = list(df.columns.difference([ 'Label', 'Attack']))
y_col = 'Label'
X = df[X_cols].values.astype('float32')
y = df[y_col].values.astype('int64')

# 划分训练集和测试集
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.4, random_state=0)

print(X_train2.shape, y_test2.shape)

X_train1 = np.load('data/train_tonb.npy')  # (175341, 32)
X_test1 = np.load('data/test_tonb.npy')  # (175341, 10)

y_train1 = np.load('data/bot_train_blabel.npy')
y_test1 = np.load('data/bot_test_blabel.npy')

y_train = np.argmax(y_train1, axis=1)
y_test = np.argmax(y_test1, axis=-1)
X_train1 = pd.DataFrame(X_train1)
X_train2 = pd.DataFrame(X_train2)

X_test1 = pd.DataFrame(X_test1)
X_test2 = pd.DataFrame(X_test2)
X_train = pd.concat([X_train2, X_train1], axis=1)
X_test = pd.concat([X_test2, X_test1], axis=1)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



train_dataset = BotDataset(X_train, y_train2)
test_dataset = BotDataset(X_test, y_test2)
#print(X_train.shape,y_train.shape)
# 构建gMLP模型
# 这里只是一个示例，你需要根据实际情况来定义你的模型

class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj_i = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj_ii = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.channel_proj_i(x)
        x = nn.functional.gelu(x)
        x = self.sgu(x)
        x = self.channel_proj_ii(x)
        return residual + x


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1,bias=True)# 44,44,1

 #       nn.init.constant_(self.spatial_proj.bias, 1.0)
    def forward(self, x):
#        residual = x
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        
        return u * v


class gMLP(nn.Module):
    def __init__(self, num_classes, d_model, d_ffn, seq_len, depth):
        super().__init__()
        self.embedding = nn.Linear(seq_len, d_model)
        self.gmlp_blocks = nn.ModuleList(
            [gMLPBlock(d_model=d_model, d_ffn=d_ffn, seq_len=seq_len) for _ in range(depth)]
        )
        self.fc = nn.Linear(depth * seq_len, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.gmlp_blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        logits = self.fc(x)

        # Apply log_softmax to the output to obtain log-probabilities
        # This is required for the input to the KLDivLoss
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs

#自定义的参数 
num_classes = #看具体的数据集种类
d_model = 
d_ffn = 
seq_len = 
depth = 
epochs= 

model = gMLP(num_classes, d_model, d_ffn, seq_len, depth).to(device)
#model= gMLP(input_size=X_train2.shape[1], hidden_size=128, num_classes=2).to(device)
#criterion = nn.CrossEntropyLoss().to(device)
# 训练模型
# 这里只是一个示例，你需要根据实际情况来定义你的训练过程
learning_rate = 0.001
# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.KLDivLoss(reduction='batchmean')
#train_dataset = BotDataset(X_train, y_train)
#test_dataset = BotDataset(X_test, y_test)
# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

#accu =Accuracy(task='binary').cuda()
#pre = Precision(task='binary').cuda()
#recall =

# 开始训练
for epoch in range(epochs):  # 假设我们训练100个epoch
    Loss=[]

    for X_batch, y_batch in train_loader:

        # 前向传播
        X_batch = X_batch.unsqueeze(1).cuda()
        #X_batch = X_batch.cuda()
        #print(X_batch.shape)
        outputs = model(X_batch).cuda()
        #outputs = torch.squeeze(outputs)
        #predictions = torch.argmax(outputs, dim=1)
        #y_batch1 = y_batch.long().cuda()
        y_batch_one_hot = nn.functional.one_hot(y_batch.long(), num_classes=2).float().cuda()
        loss = criterion(outputs, y_batch_one_hot)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss = loss.item()

    print(f'Epoch: {epoch + 1}/{epochs} | loss: {Loss: .4f}' )


    true_labels = []
    pred_labels= []

    with torch.no_grad():
      for X_batch, y_batch in test_loader:
        X_batch = X_batch.unsqueeze(1).to(device)
        import timeit

        start_time = timeit.default_timer()
        outputs = model(X_batch).to(device)
        outputs = torch.squeeze(outputs)
       # _, predicted = torch.max(outputs.data, 1)
        elapsed = timeit.default_timer() - start_time
        predicted = torch.argmax(outputs, dim=1)
       
       #
        # 计算精确率
        val_pre = precision_score(y_batch.cpu(), predicted.cpu(), average='binary')
        # 计算召回率
        recall = recall_score(y_batch.cpu(), predicted.cpu(), average='binary')
        F1 = f1_score(y_batch.cpu(), predicted.cpu(), average='binary')
        val_acc = accuracy_score(y_batch.cpu(), predicted.cpu())
        FAR = calculate_FAR(y_batch.cpu(), predicted.cpu())
      #  AUC = roc_auc_score()
        # 计算准确率
        #val_acc = calculate_class_accuracies(true_labels, pred_labels)
    print('test accuracy:', val_acc)
    print('test precision:', val_pre)
    print('recall:', recall)
    print('F1-score:', F1)
    print(str(elapsed) + ' seconds')
    print('FAR:', FAR)
