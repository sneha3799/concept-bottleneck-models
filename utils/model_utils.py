from torch import nn

def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x