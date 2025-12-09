import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


# データセットの前処理を定義
ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

# データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='data',
    train=True, #訓練用を指定
    download=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,   # テスト用を指定
    download=True,
    transform=ds_transform
)

# ミニバッチに分割する
batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

# モデルのインスタンスを作成
model = models.MyModel()

# 損失計算（誤差関数・ロス関数）の選択
loss_fn=torch.nn.CrossEntropyLoss()

# 最適化の方法の選択
learning_rate=1e-3  #学習率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs=20

train_loss_log = []
val_loss_log=[]
train_acc_log=[]
accuracy_log = []

for epoch in range(n_epochs):
    print(f'epoch {epoch+1}/{n_epochs}')

    train_loss=models.train(model,dataloader_train,loss_fn,optimizer)
    print(f'    training loss: {train_loss}')
    train_loss_log.append(train_loss)

    val_loss=models.test(model, dataloader_test, loss_fn)
    print(f'    validation loss: {val_loss}')
    val_loss_log.append(val_loss)

    train_acc = models.test_accuracy(model, dataloader_train)
    print(f'    training accuracy: {train_acc*100:.3f}%')
    train_acc_log.append(train_acc)

    accuracy=models.test_accuracy(model,dataloader_test)
    print(f'    validation accuracy: {accuracy*100:.3f}%')
    accuracy_log.append(accuracy)
    

#グラフを表示

plt.subplot(121)
plt.plot(range(1,n_epochs+1),train_loss_log)
plt.plot(range(1,n_epochs+1),val_loss_log)
plt.legend(["train","validation"])
plt.xlabel('epochs')
plt.xticks(range(1,n_epochs+1))
plt.ylabel('loss')
plt.grid()

plt.subplot(122)
plt.plot(range(1,n_epochs+1),accuracy_log)
plt.plot(range(1,n_epochs+1),train_acc_log)
plt.legend(["train","validation"])
plt.xlabel('epochs')
plt.xticks(range(1,n_epochs+1))
plt.ylabel('accuracy')
plt.grid()
plt.show()
