def accuracy(output, label, top_k=(1,)):
    max_k = max(top_k)
    batch_size = label.size(0)

    # 获取前K的索引
    _, pred = output.topk(max_k, 1, True, True)  #使用top_k来获得前k个的索引
    pred = pred.t()  # 进行转置
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred))  # 与正确标签序列形成的矩阵相比，生成True/False矩阵
    #print(correct)

    rtn = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0)  # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size))  # mul_() tensor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn


# using class to save and update accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
