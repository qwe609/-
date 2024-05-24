import torch
import Accuracy
import tqdm


def validate(validation_loader, device, model, criterion):
    model = model.to(device)  # model --> GPU
    model = model.eval()  # set eval mode
    with torch.no_grad():  # network does not update gradient during evaluation
        val_top1 = Accuracy.AverageMeter()
        validate_loader = tqdm.tqdm(validation_loader)
        validate_loss = 0
        for i, data in enumerate(validate_loader):
            inputs, labels = data[0].to(device), data[1].to(device)  # data, label --> GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            pred1, pred2 = Accuracy.accuracy(outputs, labels, top_k=(1, 2))
            n = inputs.size(0)  # batch_size=32
            val_top1.update(pred1.item(), n)
            validate_loss += loss.item()
            postfix = {'validation_loss': '%.6f' % (validate_loss / (i + 1)), 'validation_acc': '%.6f' % val_top1.avg}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return val_acc
