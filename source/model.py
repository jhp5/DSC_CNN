import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from d2l import torch as d2l
import argparse
import datetime
import wandb 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='DSC_CNN_demo01', help='project_name')
parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--optim_type', type=str, default='Adam', help='Optimizer')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--model_pth_name', type=str, default='../save/model_default.pth', help='pth_name')
# 不会解析命令行参数
args = parser.parse_args()
# 返回一个NameSpace对象

def min_max_normalize(data, dim=-1):
    """
    对输入数据在指定维度上进行 Min - Max 归一化
    """
    min_vals = data
    min_vals, _ = torch.min(min_vals, dim=dim, keepdim=True)
    max_vals = data
    max_vals, _ = torch.max(max_vals, dim=dim, keepdim=True)
    
    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1e-8
    
    normalized_data = (data - min_vals) / denominator
    return normalized_data

def create_input_iamge(data, length):
    num = len(data)//(length**2)
    data = data[:num*(length**2)].reshape((num, 1, length, length))
    data = torch.tensor(data, dtype=torch.float32)
    # Max-Min 归一化
    data = min_max_normalize(data, dim=-1)
    return data

def create_dataloaders(length, Data_names, Data_names_key, Labels, args = args):
    for i in range(len(Data_names)):
        print('正在读取：' + Data_names[i])
        mat_data = scipy.io.loadmat('../data/' + Data_names[i])
        numpy_X = mat_data[Data_names_key[i]]
        print('ndarry 形状：', numpy_X.shape)
        tensor_X = create_input_iamge(numpy_X, length)
        tensor_y = torch.full((tensor_X.shape[0],), Labels[i], dtype=torch.int64)
        print('tensor 形状：', tensor_X.shape)
        if i==0:
            X = tensor_X
            y = tensor_y
        else:
            # dim是轴而不是方向i
            X = torch.cat([X, tensor_X], dim=0)
            y = torch.cat([y, tensor_y], dim=0)
    print('模型输入的 tensor 形状为：', X.shape)
    print('对应标签的 tensor 形状为：', y.shape)
    
    dataset = TensorDataset(X, y)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    # 随机73划分数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    return train_dataloader, val_dataloader

# 定义深度可分离卷积层
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 定义Fire模块
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.squeeze(x)
        x = torch.relu(x)
        return torch.cat([
            torch.relu(self.expand1x1(x)),
            torch.relu(self.expand3x3(x))
        ], 1)

# 定义卷积神经网络
class DSC_CNN(nn.Module):
    def __init__(self):
        super(DSC_CNN, self).__init__()
        self.layer1 = DepthwiseSeparableConv(1, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire = FireModule(32, 16, 32)
        self.concat_conv = DepthwiseSeparableConv(64, 64) # 64 because FireModule output is concatenated
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5) # 推理时默认不会调用
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 4)
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.fire(x)
        x = self.concat_conv(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # 改动1：避免softmax重复计算
        # x = self.softmax(x) 
        return x
    
def train_epoch(model, dataloader, optimizer):
    model.train()
    for step, batch in enumerate(dataloader):
        features, labels = batch
        features, labels = features.to(device),labels.to(device)

        preds = model(features)
        loss = nn.CrossEntropyLoss()(preds,labels)        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return loss

def eval_epoch(model, dataloader):
    model.eval()
    accurate = 0
    num_elems = 0
    for batch in dataloader:
        features,labels = batch
        features,labels = features.to(device),labels.to(device)
        with torch.no_grad():
            preds = model(features)
        predictions = preds.argmax(dim=-1)
        accurate_preds = (predictions==labels)
        num_elems += accurate_preds.shape[0]
        accurate += accurate_preds.long().sum()
        
    val_acc = accurate.item() / num_elems
    loss = nn.CrossEntropyLoss()(preds, labels)# 不完全的
    return val_acc, loss

def train(args = args):
    Data_names = ['normal_0', 'IR007_0', 'B007_0', 'OR007@6_0', 'OR007@3_0', 'OR007@12_0']
    Data_names_key = ['X097_FE_time', 'X278_FE_time', 'X282_FE_time', 'X294_FE_time', 'X298_FE_time', 'X302_FE_time']
    Labels = [0, 1, 2, 3, 3, 3]
    train_dataloader, val_dataloader = create_dataloaders(32, Data_names, Data_names_key, Labels)
    
    model = DSC_CNN()
    model.to(device)
    optimizer = torch.optim.__dict__[args.optim_type](params=model.parameters(), lr=args.lr)
    #======================================================================
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=args.project_name, config = args.__dict__, name = nowtime, save_code=True)
    model.run_id = wandb.run.id
    #======================================================================    
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1,args.epochs+1):
        train_loss = train_epoch(model, train_dataloader, optimizer)
        val_acc, val_loss = eval_epoch(model, val_dataloader)
        train_loss_list.append(train_loss.item())
        val_loss_list.append(val_loss.item())  
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"epoch【{epoch}】@{nowtime} --> val_acc= {100 * val_acc:.2f}%")
        
        #======================================================================
        wandb.log({'train_loss':train_loss, 'val_loss':val_loss, 'val_acc': val_acc})
        #======================================================================        
    # 这个画图函数源自 d2l，只能在 jupyter 中画图
    # 所以使用 wand 监视损失变化
    # d2l.plot(list(range(1, args.epochs + 1)), [train_loss_list, val_loss_list],xlabel='epoch',
    #          ylabel='loss', xlim=[1, args.epochs], ylim=[0, 2],
    #          legend=['train_loss', 'valid_loss'], yscale='linear')
    torch.save(model.state_dict(), args.model_pth_name)
    wandb.finish()
    
train(args)