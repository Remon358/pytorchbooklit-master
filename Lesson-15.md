Lesson-15 

RNN 循环神经网络

主要用于处理序灌问题(输入数据之间存在一定的相关性)

下面给出一个学习实例, 主要完成的输入数据为"hello" -->"ohlol"

```python
import torch
batch_size = 1
seq_len = 5
input_size = 4
hidden_size = 4
num_layer = 1

idx2char = ['e', 'h', 'l', 'o']

x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)

class NLPModel(torch.nn.Module):
    def __init__(self):
        super(NLPModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layer)
    def forward(self, x):
        hidden = torch.zeros(num_layer, batch_size, hidden_size)
        out, _ = self.rnn(x, hidden)
        return out.view(-1, hidden_size)

model = NLPModel()
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(35):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optim.zero_grad()
    loss.backward()
    optim.step()

    _, idex = outputs.max(dim= 1)
    idx = idex.data.numpy()
    print('Predicted:', ''.join([idx2char[x] for x in idx]), end='')
    print(f'\t epoch={epoch}, loss={loss.item()}')
```