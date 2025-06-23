class RVTDNN(torch.nn.Module):
    def __init__(self, n_input, n_hidden1, n_output):
        super(RVTDNN, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden1)
        self.activation = torch.nn.Tanh()
        self.output = torch.nn.Linear(n_hidden1, n_output)
    def forward(self, x):
        out = self.hidden1(x)
        out = self.activation(out)
        out = self.output(out)
        return out
