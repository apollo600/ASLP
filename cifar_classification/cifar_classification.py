import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import traceback

from testNet import testNet
from pic import pic
from convNet import convNet

import os


def train(trainloader, epoches, save_parameter, resume_last):

    # 初始化
    net = convNet()
    net.to(device)
    if resume_last:
        try:
            net.load_state_dict(torch.load(SAVE_PATH))
        except:
            print("[ERROR] no model saved in", SAVE_PATH)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.00001)
    # optimizer = optim.Adam(net.parameters(), lr = 0.0001, weight_decay=0.00001)
    batch_size = len(trainloader)

    print("========================")
    print("\nTrain Begin!\n")
    print("[INFO] using NET:", net, "\n")
    
    for epoch in range(epoches):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # 所有图像及其标签
            inputs, labels = data[0].to(device), data[1].to(device)
            # 优化器重置
            optimizer.zero_grad()
            # 开训
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 输出结果
            running_loss += loss.item()
            if i % 2000 == 1999: # 每2000个batch输出一次
                loss = running_loss / 2000
                train_acc = test_fromNet(net, trainloader)
                test_acc = test_fromNet(net, testloader)
                this_epoch = epoch+(i+1)/batch_size
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f} train_acc: {train_acc:.4f} test_acc: {test_acc:.4f}')
                respic.add([this_epoch, this_epoch, this_epoch], [loss, train_acc, test_acc])
                running_loss = 0.0
            
        
    print("\nTrain Finshed!\n")
    respic.draw()
    if save_parameter:
        torch.save(net.state_dict(), SAVE_PATH)
        print("Parameter Saved to", SAVE_PATH, "\n")
    print("========================")
    

def test_fromNet(net, testloader):

    correct = 0
    total = 0
    with torch.no_grad():

        for data in testloader:

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            
            # 得出正确率
            total += labels.size(0)
            correct += (predictions == labels).sum().item()   
    
    try:    
        # print(f'Accuracy of the network: {100 * correct // total} %')
        return float(correct) / total
    except:
        print("[ERROR] no data in dataloader!")
        traceback.print_exc()


def test_fromModel(testloader, classes):
    
    net = convNet()
    net.to(device)
    try:
        net.load_state_dict(torch.load(SAVE_PATH))
    except:
        print("[ERROR] no model saved in", SAVE_PATH)
    
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    accuracy_pred = {classname: 0.0 for classname in classes}

    with torch.no_grad():

        for data in testloader:

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            
            # 统计结果
            for label, prediction in zip(labels, predictions):

                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            
        for classname, correct_count in correct_pred.items():

            try:
                accuracy = float(correct_count) / total_pred[classname]
                accuracy_pred[classname] = accuracy
                print(f'Accuracy for class: {classname:5s} is {100 * accuracy:.1f} %')
            except:
                print("[ERROR] no data of", classname, "in dataloader")
                
        return accuracy_pred


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    SAVE_PATH = '/home/disk1/user2/mxy/cifar_classification/cifar_net.pth'

    trainset = torchvision.datasets.CIFAR10(root='/home/disk1/user2/mxy/cifar_classification/data', train=True,
                                        download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='/home/disk1/user2/mxy/cifar_classification/data', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    classes = trainset.classes

    respic = pic("epoches", "", ["loss", "train_acc", "test_acc"])

    print("[INFO] use device: ", device)
    print("[INFO] train dataset: ", trainset)
    print("[INFO] test dataset: ", testset)
    print("[INFO] classes: ", classes)

    oom = False
    try:
        train(trainloader, 5, save_parameter=True, resume_last=True)
    except RuntimeError:
        oom = True
    if oom:
        train(trainloader, 5, save_parameter=True, resume_last=True)
    test_fromModel(testloader, classes)
