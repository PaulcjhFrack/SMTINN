# batch_size learning
def Batch_size(train_x, train_y, batch_size):
    # from numpy array to torch tensor
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    train_xy = TensorDataset(train_x, train_y)

    train_loader = torch.utils.data.DataLoader(train_xy, batch_size=batch_size, shuffle=True)

    return train_x, train_y, train_loader
