# main function to be executed
import torch
import yaml
from torch import nn, optim
from torchvision import models
from Prepare import MyDataset
import Prepare
from Submission import submission
from train_fn import train
from Validate_fn import validate


def main():
    # set hyper-parameter of train, eval scripts
    with open("config.yaml", encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        epochs = cfg['epochs']
        lr = cfg['lr']
        batch_size = cfg['batch_size']
        train_path = cfg['train_path']
        test_path = cfg['test_path']
        csv_path = cfg['csv_path']
        tensorboard_path = cfg['tensorboard_path']
        model_save_path = cfg['model_save_path']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # using customized network
    # net = MyCNN()

    # using ResNet18 network and pretrained weights (network need fine-tune)
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(2048, 2)  # adjust the output of last layer to 2 dim
    net = resnet50

    # using ResNet50 network and pretrained weights (network need fine-tune)
    # resnet50 = models.resnet50(pretrained=True) # automatically download weights from Internet if no weights in cache
    # resnet50.fc = nn.Linear(2048, 2) # adjust the output of last layer to 2 dim
    # net = resnet50

    train_ds = MyDataset(train_path)
    new_train_ds, validate_ds = Prepare.MyDataset.dataset_split(train_ds, 0.8)
    test_ds = MyDataset(test_path, train=False)

    new_train_loader = Prepare.dataloader(new_train_ds, batch_size)
    validate_loader = Prepare.dataloader(validate_ds, batch_size)
    test_loader = Prepare.dataloader(test_ds, batch_size)

    criterion = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
    # execute train function
    train(new_train_loader, device, net, epochs, lr, criterion, optimizer, tensorboard_path)
    # save trained model
    torch.save(net.state_dict(), model_save_path)
    # execute evaluation function
    # 1.load model, load parameter
    val_net = resnet50
    val_net.load_state_dict(torch.load(model_save_path))
    # 2.execute evaluation function
    val_acc = validate(validate_loader, device, val_net, criterion)
    print('val_acc:', '%.2f' % val_acc + "%")
    # 3.generate csv_path
    submission(csv_path, test_loader, device, net)


if __name__ == '__main__':
    main()
