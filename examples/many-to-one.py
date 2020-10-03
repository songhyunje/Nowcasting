import torch
from torch import nn

from ConvLSTM import ConvLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # define batch_size, channels, height, width
    batch_size, channels, height, width = 24, 10, 100, 100
    d = 1  # hidden state size
    lr = 1e-3  # learning rate
    sequence_length = 6  # sequence length
    max_epoch = 100  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    model = ConvLSTM(10, 1, (3, 3), 3, return_all_layers=False).to(device)
    print(model)

    print('Create input and target')
    x = torch.rand(batch_size, sequence_length, channels, height, width).to(device)
    y = torch.randn(batch_size, d, height, width).to(device)
    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epoch):
        outputs, _ = model(x)
        loss = criterion(outputs[:, -1, :, :, :], y)
        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch + 1), loss.item()))

        model.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
