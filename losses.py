import torch
import torch.nn.functional as F
import utils

# def dice_loss_d(predict,
#                 target,
#                 epsilon=0.00001,
#                 weight=None,
#                 ignore_index=255):
#     """Dice loss, reduction is mean
#   Args:
#       predict: A tensor of shape [N,C,W,H]
#       target: A tensor of shape, processed by one_hot [N,W,H]
#       epsilon: epsilon, default:0.00001
#   Return:
#       mean(loss)
#   """
#     assert predict.dim() == 4, "predict dimensions must mach (N,C,H,W)"
#     assert target.dim() == 3, "target dimensions must mach (N,H,W)"
#     assert weight == None or isinstance(
#         weight,
#         (list, tuple, torch.Tensor
#          )), "weight type error, expected (list, tuple, torch.Tensor) or None"

#     classes_num = predict.shape[1]
#     reduce_index = (1, 2, 3)

#     if weight is None:
#         weight = torch.ones(1, classes_num)
#     elif isinstance(weight, (list, tuple, torch.Tensor)):
#         assert len(weight) == classes_num
#         weight = torch.FloatTensor(weight).view(1, -1)
#         weight = weight / weight.sum()
#     weight.to(target.device)
#     target = utils.one_hot_encode(target, classes_num)
#     predict = F.softmax(predict, 1)
#     inse = (torch.sum(predict * target, (2, 3)) * weight).sum(1)
#     denominator = torch.sum(predict, reduce_index) + torch.sum(
#         target, reduce_index)
#     dice_score = 1 - inse * 2 / (denominator + epsilon)
#     return torch.mean(dice_score)

# def focal_loss_d(predict, target, epsilon=0.00001, alpha=None, gamma=2):
#     """focal loss, reduction is mean       (discard)
#   Args:
#       predict: A tensor of shape [N,C,W,H]
#       target: A tensor of shape, processed by one_hot [N,W,H]
#   Return:
#       mean(loss)
#   """
#     assert predict.dim() == 4, "predict dimensions must mach (N,C,H,W)"
#     assert target.dim() == 3, "target dimensions must mach (N,H,W)"

#     classes_num = predict.shape[1]
#     reduce_index = (1, 2, 3)

#     if alpha is None:
#         alpha = torch.ones(1, classes_num, 1, 1)
#     elif isinstance(alpha, (list, tuple, torch.Tensor)):
#         assert len(alpha) == classes_num
#         alpha = torch.FloatTensor(alpha).view(1, -1, 1, 1)
#         alpha = alpha / alpha.sum()
#     alpha.to(target.device)

#     target = utils.one_hot_encode(target, classes_num)
#     predict = F.softmax(predict, 1)

#     alpha = (target * alpha).sum(1)
#     probs = (predict * target).sum(1)  # shape [N,1]

#     loss = (torch.pow(1 - probs, gamma)) * probs.log()
#     loss = -alpha * loss
#     return loss.mean()


def focal_loss(predict,
               target,
               epsilon=0.00001,
               weight=None,
               gamma=2,
               ignore_index=255):
    """focal loss, reduction is mean  
    Args:
        predict: A tensor of shape [N,C,W,H]
        target: A tensor of shape, processed by one_hot [N,W,H]
    Return:
        mean(loss)
    """
    assert predict.dim() == 4, "predict dimensions must mach (N,C,H,W)"
    assert target.dim() == 3, "target dimensions must mach (N,H,W)"

    classes_num = predict.shape[1]
    reduce_index = (1, 2, 3)

    predict = F.softmax(predict, 1)
    print("[debug-afterSoftMax]".format(predict.unique()))
    print("[debug-log]".format(predict.log()))
    input = torch.pow((1 - predict), gamma) * predict.log()

    return F.nll_loss(input, target, weight=weight, ignore_index=ignore_index)


def dice_loss(predict, target, epsilon=0.00001, weight=None, ignore_index=255):
    """Dice loss, reduction is mean
  Args:
      predict: A tensor of shape [N,C,W,H]
      target: A tensor of shape, processed by one_hot [N,W,H]
      epsilon: epsilon, default:0.00001
  Return:
      mean(loss)
  """
    assert predict.dim() == 4, "predict dimensions must mach (N,C,H,W)"
    assert target.dim() == 3, "target dimensions must mach (N,H,W)"

    classes_num = predict.shape[1]
    w, h = predict.shape[2:]
    reduce_index = (1, 2, 3)

    # onehot = utils.one_hot_encode(target, classes_num)
    predict = F.softmax(predict, 1)

    # denominator = torch.sum(predict, reduce_index) + torch.sum(
    #     onehot, reduce_index) + epsilon
    denominator = 2 * torch.sum(predict, reduce_index) + epsilon

    # nll函数出来的值已经取过负值
    dice = F.nll_loss(predict * w * h / denominator.view(-1, 1, 1, 1),
                      target,
                      weight=weight,
                      ignore_index=ignore_index)

    return 1 + 2 * dice


# if __name__ == "__main__":
# import numpy as np
# #------------------------------------------------------
# predict = torch.tensor(np.random.rand(4, 8, 10, 10))
# target = torch.tensor(
#     np.random.randint(0, 8, 400, dtype=np.int64).reshape((4, 10, 10)))
# result1 = dice_loss(predict, target)
# result2 = dice_loss_d(predict, target)
# print(result1)
# print(result2)
#------------------------------------------------------
# predict = torch.tensor(np.random.rand(4, 8, 10, 10))
# target = torch.tensor(
#     np.random.randint(0, 8, 400, dtype=np.int64).reshape((4, 10, 10)))
# alpha = [75 / 7] * 8
# alpha[0] = 25
# result = focal_loss(predict, target, alpha=(alpha))
# result = focal_loss(predict, target)

# print(result)
