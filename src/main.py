import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from net import CNN
from data import CreateDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True


train_data = np.loadtxt(
    fname="../input/train.csv",
    delimiter=",",
    skiprows=1
)
test_data = np.loadtxt(
    fname="../input/test.csv",
    delimiter=",",
    skiprows=1
)
n_train = int(train_data.shape[0])
n_test = int(test_data.shape[0])


train_x = (train_data[:, 1:]).reshape(-1, 1, 28, 28).astype(np.float32)
train_y = (train_data[:, [0]]).reshape(-1).astype(np.int64)
test_x = test_data.reshape(-1, 1, 28, 28).astype(np.float32)


train_set = CreateDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)


net = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=0.0005,
    momentum=0.99,
    nesterov=True
)


epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


submission = np.zeros((n_test, 2))
net.eval()
for i in range(n_test):
    data = torch.from_numpy(test_x[i].reshape(1, 1, 28, 28)).to(device)
    output = net(Variable(data))
    submission[i][0] = i + 1
    submission[i][1] = torch.argmax(output.data)

print('Finished Evaluation')


np.savetxt(
    fname="../output/submission.csv",
    X=submission,
    fmt="%.0f",
    delimiter=",",
    header="ImageId,Label",
    comments=""
)

print('Saved the Results')
