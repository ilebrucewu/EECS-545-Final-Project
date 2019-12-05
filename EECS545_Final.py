import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.25 for _ in range(10)], gamma=0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.Tensor(alpha)
        self.gamma = gamma
		
    def forward(self, input, target):
        # one-hot encoding
        y = torch.zeros([target.size(-1), input.size(-1)])
        for i, t in enumerate(target):
            y[i, t] = 1 

        # Use alpha to adjust the weight of loss
        self.alpha = self.alpha.type_as(input.data)
        coeff = self.alpha.gather(0, target.data.view(-1)).unsqueeze(1).repeat(1, 10)

        pt = F.softmax(input, dim=-1).clamp(1e-7, 1 - 1e-7)
        ce_loss = -1 * y * (coeff * torch.log(pt)) # weighted cross entropy
        f_loss = ce_loss * (1 - pt) ** self.gamma  # focal loss
        
        return f_loss.sum() / target.size(-1)             

# Model (Num of classes = 10)
class ConvNet(nn.Module):
    def __init__(self, chs=1, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chs, 32, kernel_size=3, stride=1, padding=1),#nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16 * 6 * 6, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x       

# MNIST dataset
def MNIST_loader(ratio = [1.0 for _ in range(10)], special_class=[]):
    """ Load the MNIST dataset (from Pytorch), return train_loader & test_loader
	Number of training data = 60000, Number of testing data = 10000, batch size = 100

    Args:
        ratio: the proportion of data to use for every classes, len(ratio) = 10
		special_class: Generate testing data from specific classes

    Returns:
        train_loader: DataLoader of training data: (x, target)
        test_loader:  DataLoader of testing data:  (x, target)
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    batch_size = 100
    train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
    test_set  = dset.MNIST(root='./data', train=False, transform=trans, download=True)
    
    train_set.targets = torch.tensor(train_set.targets)
    # Generating new training dataset (Unbalanced)
    for i in range(10):
        if(i == 0):
            idx = train_set.targets == 0
            data_class, target_class = train_set.data[idx.numpy().astype(np.bool)], train_set.targets[idx]
            data_new = data_class[:int(len(target_class) * ratio[i]),:,:]
            target_new = target_class[:int(len(target_class) * ratio[i])]
        else:
            idx = train_set.targets == i
            data_class, target_class = train_set.data[idx.numpy().astype(np.bool)], train_set.targets[idx]

            data_new = torch.cat([data_new, data_class[:int(len(target_class) * ratio[i]),:,:]], dim = 0)
            target_new = torch.cat([target_new, target_class[:int(len(target_class) * ratio[i])]], dim = 0)
    
    # Last step: Append the new data & target to the original set
    train_set.targets= target_new
    train_set.data = data_new

    # Generate special testcases (Extract those classes from testing data)
    if(special_class):
        test_set.targets = torch.tensor(test_set.targets)
        for i in range(len(special_class)):
            if(i == 0):
                idx = test_set.targets == special_class[0]
                data_class, target_class = test_set.data[idx.numpy().astype(np.bool)], test_set.targets[idx]
                data_new = data_class
                target_new = target_class
            else:
                idx = test_set.targets == special_class[i]
                data_class, target_class = test_set.data[idx.numpy().astype(np.bool)], test_set.targets[idx]
                
                data_new = torch.cat([data_new, data_class], dim = 0)
                target_new = torch.cat((target_new, target_class), dim = 0)

        test_set.targets = target_new
        test_set.data = data_new
	
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# CIFAR dataset
def CIFAR_loader(ratio = [0.1 for _ in range(10)], special_class=[]):
    """ Load the CIFAR10 dataset (from Pytorch), return train_loader & test_loader
	Number of training data = 50000, Number of testing data = 10000, batch size = 100
    
    Args:
        ratio: the proportion of data to use for every classes, len(ratio) = 10
		special_class: Generate testing data from specific classes

    Returns:
        train_loader: DataLoader of training data: (x, target)
        test_loader:  DataLoader of testing data:  (x, target)

    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 100
    train_set = dset.CIFAR10(root='./data', train=True, transform=trans, download=True)
    test_set  = dset.CIFAR10(root='./data', train=False, transform=trans, download=True)

    train_set.targets = torch.tensor(train_set.targets)

    for i in range(10):
        if(i == 0):
            idx = train_set.targets == 0
            data_class, target_class = train_set.data[idx.numpy().astype(np.bool)], train_set.targets[idx]
            data_new = torch.from_numpy(data_class[:int(len(target_class) * ratio[i]),:,:,:])
            target_new = target_class[:int(len(target_class) * ratio[i])]
        else:
            idx = train_set.targets == i
            data_class, target_class = train_set.data[idx.numpy().astype(np.bool)], train_set.targets[idx]

            data_new = torch.cat((data_new, torch.from_numpy(data_class[:int(len(target_class) * ratio[i])])), dim = 0)
            target_new = torch.cat([target_new, target_class[:int(len(target_class) * ratio[i])]], dim = 0)

    # Last step: Append the new data & target to the original set
    train_set.targets = target_new
    train_set.data = data_new.numpy()

    # Generate special testcases (Extract those classes from testing data)
    if(special_class):
        test_set.targets = torch.tensor(test_set.targets)
        for i in range(len(special_class)):
            if(i == 0):
                idx = test_set.targets == special_class[0]
                data_class, target_class = test_set.data[idx.numpy().astype(np.bool)], test_set.targets[idx]
                data_new = torch.from_numpy(data_class)
                target_new = target_class
            else:
                idx = test_set.targets == special_class[i]
                data_class, target_class = test_set.data[idx.numpy().astype(np.bool)], test_set.targets[idx]
                
                data_new = torch.cat((data_new, torch.from_numpy(data_class)), dim = 0)
                target_new = torch.cat((target_new, target_class), dim = 0)

        test_set.targets = target_new
        test_set.data = data_new.numpy()

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def k_fold_DataLoader(batch_num, k = 5):
    """Generate the batch idx which should be used for validation (for dataset using torch.utils.data.DataLoader)
    
    Args:
        batch_num: # of batch (batch_num = num_of_data / batch_size)

    Returns:
        batch_val: list of list (each list contains the batch selected to validate the data)
    """
    divide = int(batch_num / k)
    idx_list = []
    for i in range(k):
        idx_list.append([j for j in range(divide * i, min(divide * (i + 1), batch_num))])
    return idx_list

# training
def training(train_loader, test_loader, model, epochs = 400, criterion = nn.CrossEntropyLoss()):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    torch.manual_seed(0)
    acc = []
    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

        if((batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader)):
            print('>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx+1, loss.item()))
        
        correct_cnt, total_cnt = 0, 0  # Used for calculating the precision of the testing result
        if((epoch + 1) % 5 == 0):
            for batch_idx, (x, target) in enumerate(test_loader):
                out = model(x)
                _, pred_label = torch.max(out.data, 1)
                total_cnt += x.data.size()[0]
                correct_cnt += (pred_label == target.data).sum()
            
            if(batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
                print('>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.6f}'.format(epoch, batch_idx+1, loss.item(), correct_cnt * 1.0 / total_cnt))
            acc.append(correct_cnt * 1.0 / total_cnt)
    
    return model, acc
	
def confusion_matrix(test_loader, model):
	model.eval() # Set the model to evaluation mode
	
	for batch_idx, (x, target) in enumerate(test_M):
		out = model(x)
		_, pred_label = torch.max(out.data, 1)
		
		if(batch_idx == 0):
			y_predict = pred_label
			y_true = target
		else:
			y_predict = torch.cat((y_predict, pred_label), dim = 0)
			y_true = torch.cat((y_true, target), dim = 0)
			
	return confusion_matrix(y_true, y_predict)

def main():
	# For imbalanced dataset, we need to pass ratio, special_class as an argument to the data loader function
	# We also need to pass alpha to the focal loss function
	ratio = [0.3, 0, 0, 0, 0.3, 0.3, 0, 0, 0.05, 0.05]
	alpha = [0.2/3, 0, 0, 0, 0.2/3, 0.2/3, 0, 0, 0.4, 0.4] # Inverse of frequency of data
	special_class = [0, 4, 5, 8, 9]
	train_C, test_C = CIFAR_loader(ratio=ratio, special_class=special_class)
	
	model = ConvNet(chs = 3) 	# Channel == 3 for CIFAR10
	model_new, acc = training(train_C, test_C, model, criterion = FocalLoss(gamma=2.0, alpha=alpha))
	print(confusion_matrix(test_C, model_new))

if __name__ == "__main__":
	main()
