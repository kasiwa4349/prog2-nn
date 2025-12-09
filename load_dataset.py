import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms


# データセットを読み込む
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True
)

print(f'dataset size: {len(ds_train)}')

# インデックスを指令して、データを取り出す
# データ（画像）とターゲット（クラス番号）のタプル
image, target = ds_train[0]

print(type(image))
print(target)

#表示
plt.imshow(image, cmap='gray_r',vmin=0,vmax=255)
plt.title(target)
plt.show()

# PIL -> torch.Tensor
image = transforms.functional.to_image(image)
print(image.shape, image.dtype)

#PIL画像をtorch.tensorに変換する
image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image, dtype=torch.float32,scale=True)
print(image.shape, image.dtype)
print(image.min(), image.max())