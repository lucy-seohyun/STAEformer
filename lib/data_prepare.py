import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange

# ! X shape: (B, T, N, C)

# 커스텀 스케일러
import pickle
class ChainedTransformer():
    def __init__(self, mean, std):
        minmax = pickle.load(open('./data/CHUNGNAM/scaler.pkl', 'rb')) # 전처리 과정에서 사용한 스케일러 읽어오기
        self.scale_ = np.float32(minmax.scale_)
        self.min_ = np.float32(minmax.min_)
        self.scaler = StandardScaler(mean=mean, std=std)
    def transform(self, X):
        X = X * self.scale_
        X = X + self.min_
        return self.scaler.transform(X)
    def inverse_transform(self, X):
        X = X - torch.Tensor(self.min_).to('cuda:0')
        X = X / torch.Tensor(self.scale_).to('cuda:0')
        return self.scaler.inverse_transform(X)

def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
):
#     data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

#     features = [0]
#     if tod:
#         features.append(1)
#     if dow:
#         features.append(2)
#     # if dom:
#     #     features.append(3)
#     data = data[..., features]

#     index = np.load(os.path.join(data_dir, "index.npz"))

#     train_index = index["train"]  # (num_samples, 3)
#     val_index = index["val"]
#     test_index = index["test"]

#     x_train_index = vrange(train_index[:, 0], train_index[:, 1])
#     y_train_index = vrange(train_index[:, 1], train_index[:, 2])
#     x_val_index = vrange(val_index[:, 0], val_index[:, 1])
#     y_val_index = vrange(val_index[:, 1], val_index[:, 2])
#     x_test_index = vrange(test_index[:, 0], test_index[:, 1])
#     y_test_index = vrange(test_index[:, 1], test_index[:, 2])

#     x_train = data[x_train_index]
#     y_train = data[y_train_index][..., :1]
#     x_val = data[x_val_index]
#     y_val = data[y_val_index][..., :1]
#     x_test = data[x_test_index]
#     y_test = data[y_test_index][..., :1]

      
    
#     x_train[..., 0] = scaler.transform(x_train[..., 0])
#     x_val[..., 0] = scaler.transform(x_val[..., 0])
#     x_test[..., 0] = scaler.transform(x_test[..., 0])
    
    xtr= np.load(os.path.join(data_dir, "train.npz"))["x"].astype(np.float32)
    xv=  np.load(os.path.join(data_dir, "val.npz"))["x"].astype(np.float32)
    xte= np.load(os.path.join(data_dir, "test.npz"))["x"].astype(np.float32)
    
    ytr= np.load(os.path.join(data_dir, "train.npz"))["y"].astype(np.float32)
    yv=  np.load(os.path.join(data_dir, "val.npz"))["y"].astype(np.float32)
    yte= np.load(os.path.join(data_dir, "test.npz"))["y"].astype(np.float32)
    
    
    x_train = xtr
    x_val=xv
    x_test=xte
    
    y_train=ytr
    y_val=yv
    y_test=yte
    
    # X, y 다 변환
    scaler = ChainedTransformer(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])
    y_train[..., 0] = scaler.transform(y_train[..., 0])
    y_val[..., 0] = scaler.transform(y_val[..., 0])
    y_test[..., 0] = scaler.transform(y_test[..., 0])
    
    
    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
