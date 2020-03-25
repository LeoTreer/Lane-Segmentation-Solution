import torch
import torch.nn.functional as F

input = torch.rand((2, 8, 10, 10))
label = torch.randint(0, 8, (2, 10, 10))

onehot = F.one_hot(label)

o = onehot.permute((0, 3, 1, 2))

output = F.nll_loss(input, label)

cul = (o * input).sum(1).mean()

print("lose output:{}, cul output:{}".format(output, cul))
