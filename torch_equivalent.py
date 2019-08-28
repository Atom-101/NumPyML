import torch.nn as nn
import glob
import numpy as np
import cv2
import torch

model = nn.Sequential(
    nn.Conv2d(3,6,5,padding=2),
    nn.ReLU(),
    nn.Conv2d(6,16,5,padding=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16384,84),
    nn.ReLU(),
    nn.Linear(84,10)
    )
opt = torch.optim.SGD(model.parameters(),1e-4)
file_names = glob.glob('MnistTrain/*/*.jpg')
labels = np.zeros(len(file_names))

for i,name in enumerate(file_names):
    labels[i] = int(name.split('/')[-2])
loss_fn = nn.CrossEntropyLoss()
for _ in range(10):
    for i in range(0,-1,8):
        X = []
        for address in file_names[i:i+8]:
            im = cv2.resize(cv2.imread(address),(32,32),interpolation=cv2.INTER_AREA)
            im = im/255.0
            # im = im.reshape(-1)
            X.append(im)
        y = model(torch.Tensor(X).permute(0,3,1,2))
        loss = loss_fn(y.squeeze(),torch.LongTensor(labels[i:i+8]))
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss)



