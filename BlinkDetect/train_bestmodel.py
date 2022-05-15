from customdataset import CustomDateset
from torchvision import transforms
import torch.utils.data as Data
from model import Lenet
import torch
import torch.nn.functional as F

kBatchSize = 32
kNumberOfEpochs = 2000
testfre = 2     # 个epoch测试一下

modelPath = "/Users/ayang/PycharmProjects/pythonProject/bestmodel.pth"

def main():
    minloss = 100
    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=(50, 50)),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(size=(50, 50)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_file_name_txt = "/Users/ayang/PycharmProjects/pythonProject/train.txt"
    test_file_name_txt = "/Users/ayang/PycharmProjects/pythonProject/test.txt"

    train_dataset = CustomDateset(train_file_name_txt, transforms=data_transforms['train'])
    test_dataset = CustomDateset(test_file_name_txt, transforms=data_transforms['val'])

    num_train_samples = train_dataset.__len__() // kBatchSize * kBatchSize
    num_test_samples = test_dataset.__len__() // kBatchSize * kBatchSize

    train_data_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=kBatchSize,
        shuffle=True,
        num_workers=7
    )

    test_data_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=kBatchSize,
        shuffle=True,
        num_workers=7
    )

    net = Lenet()
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # net.eval()
    # optimizer.zero_grad()

    for epochs in range(kNumberOfEpochs):
        net.train(mode=True)
        train_running_loss = 0
        train_running_corrects = 0
        test_running_loss = 0
        test_running_corrects = 0
        train_corrects = 0

        for batch, (image, label) in enumerate(train_data_loader):
            outputs = net(image)
            label = label.view(-1)
            label = label.long()
            loss = F.cross_entropy(input=outputs, target=label)
            train_running_loss += loss * image.size(0)

            predict = torch.max(outputs, 1)[1]
            # train_corrects = torch.sum(predict == label.data)
            # train_running_corrects += train_corrects.item()

            train_running_corrects += torch.sum(predict == label.data).item()
            # same = predict.equal(label)
            # print(train_running_corrects)

            net.eval()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sample_train_loss = train_running_loss / num_train_samples
        sample_train_accuracy = train_running_corrects / num_train_samples
        print('[{:.1f}/{:.1f}] Train Loss: {:.4f} Acc: {:.4f}'.format(epochs, kNumberOfEpochs,
                                                                      sample_train_loss, sample_train_accuracy))

        if(epochs % testfre == 0):
            for batch, (image, label) in enumerate(test_data_loader):
                outputs = net(image)
                label = label.view(-1)
                label = label.long()

                loss = F.cross_entropy(input=outputs, target=label)
                test_running_loss += loss * image.size(0)
                predict = torch.max(outputs, 1)[1]
                # train_corrects = torch.sum(predict == label.data)
                # train_running_corrects += train_corrects.item()
                test_running_corrects += torch.sum(predict == label.data).item()

            sample_test_loss = test_running_loss / num_test_samples
            sample_test_accuracy = test_running_corrects / num_test_samples
            print('[{:.1f}/{:.1f}] Test Loss: {:.4f} Acc: {:.4f}'.format(epochs, kNumberOfEpochs, sample_test_loss,
                                                                         sample_test_accuracy))

            if(minloss > sample_test_loss):
                minloss = sample_test_loss
                # torch.save(Lenet.state_dict(), modelPath)
                torch.save(net.state_dict(), modelPath)
                # torch.save(net, modelPath)
                print('save best model on epoch:{:.4f}'.format(epochs))




if __name__ == '__main__':
    main()

