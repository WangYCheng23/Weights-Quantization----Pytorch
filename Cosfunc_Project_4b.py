import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import math
from tensorboardX import SummaryWriter
from lenet5 import LeNet5


writer = SummaryWriter('runs/dissertation/CosReg4b/5')
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST(r'..\input\MNIST', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = datasets.MNIST(r'..\input\MNIST', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)

#dataiter = iter(trainloader) # creating a iterator
#images, labels = dataiter.next() # creating images for image and lables for image number (0 to 9) 

# Model creation with neural net Sequential model
model=LeNet5()
model.to(dev)


class Cos_Loss(nn.Module):
    def __init__(self, lambd = 1e-2):
        super(Cos_Loss,self).__init__()
        self.lambd = lambd

    def forward(self, net, u):
        pi = 3.1415927410125732
        loss = 0
        for param in net.parameters():
            loss = loss + (torch.cos(torch.pow(2*pi/u,(param - u/2)))+1).sum()

        return loss*self.lambd

def quantization(weights,bit,boundry_value):
    param = (1/boundry_value)*(2**bit)
    q_weights = (torch.round(param*weights))/param
    q_weights = q_weights.clamp(min=-boundry_value,max=boundry_value)
    return q_weights

criterion = nn.NLLLoss().to(dev) 

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)

time0 = time()
Iterations = 50
epochs = 10
running_loss_list= []

epochs_list = []
bit = 3
u = 1/(2**bit)
lambd_val = 1e-8

for iterations in range(Iterations):

    if iterations == 5:
        lambd_val = lambd_val*100
    # elif iterations == 10:
    #     lambd_val = lambd_val*200

    loss_func = Cos_Loss(lambd=lambd_val).to(dev)
    for e in range(epochs):
        running_loss = 0 
        for i,data in enumerate(trainloader,0):
            images, labels = data[0].to(dev), data[1].to(dev)
            # defining gradient in each epoch as 0
            optimizer.zero_grad()           
            # modeling for each image batch
            output = model(images)           
            # calculating the loss
            loss = criterion(output, labels)
            my_loss = loss_func(model,u)
            # writer.add_scalar('NLL_Loss',loss,)
            # writer.add_scalar('Cos_Loss',my_loss)
            loss = loss + my_loss           
            # This is where the model learns by backpropagating
            loss.backward()          
            # And optimizes its weights here
            optimizer.step()
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format((iterations*epochs)+e, running_loss/len(trainloader)))
            writer.add_scalar('Cos_Loss',my_loss,(iterations*epochs)+e)
            writer.add_scalar('AllLoss',running_loss,(iterations*epochs)+e)
            conv1_weight = model.c1.c1.c1.weight.to("cpu").detach().numpy()
            conv2_1_weight = model.c2_1.c2.c2.weight.to("cpu").detach().numpy()
            conv2_2_weight = model.c2_2.c2.c2.weight.to("cpu").detach().numpy()
            fullyconnect3_weight = model.c3.c3.c3.weight.to("cpu").detach().numpy()
            fullyconnect4_weight = model.f4.f4.f4.weight.to("cpu").detach().numpy()
            fullyconnect5_weight = model.f5.f5.f5.weight.to("cpu").detach().numpy()
            writer.add_histogram('Conv1_layer_weight',conv1_weight,(iterations*epochs)+e)
            writer.add_histogram('Conv2_1_layer_weight',conv2_1_weight,(iterations*epochs)+e)
            writer.add_histogram('Conv2_2_layer_weight',conv2_2_weight,(iterations*epochs)+e)
            writer.add_histogram('Fully_connect3_layer_weight',fullyconnect3_weight,(iterations*epochs)+e)
            writer.add_histogram('Fully_connect4_layer_weight',fullyconnect4_weight,(iterations*epochs)+e)
            writer.add_histogram('Fully_connect5_layer_weight',fullyconnect5_weight,(iterations*epochs)+e)


c1_weights = model.c1.c1.c1.weight
c2_1_weights = model.c2_1.c2.c2.weight
c2_2_weights = model.c2_2.c2.c2.weight
c3_weights = model.c3.c3.c3.weight
f4_weights = model.f4.f4.f4.weight
# fullyconnect5_weight = model.f5.f5.f5.weight
q_c1_weights = quantization(c1_weights,bit=bit,boundry_value=1)
q_c2_1_weights = quantization(c2_1_weights,bit=bit,boundry_value=1)
q_c2_2_weights = quantization(c2_2_weights,bit=bit,boundry_value=1)
q_c3_weights = quantization(c3_weights,bit=bit,boundry_value=1)
q_f4_weights = quantization(f4_weights,bit=bit,boundry_value=1)
model.state_dict()['c1.c1.c1.weight'].data.copy_(q_c1_weights)
model.state_dict()['c2_1.c2.c2.weight'].data.copy_(q_c2_1_weights)
model.state_dict()['c2_2.c2.c2.weight'].data.copy_(q_c2_2_weights)
model.state_dict()['c3.c3.c3.weight'].data.copy_(q_c3_weights)
model.state_dict()['f4.f4.f4.weight'].data.copy_(q_f4_weights)

# Testing
aver_accur = 0.0
for num_aver in range(10):
    correct_count, all_count = 0, 0
    for images,labels in testloader:
        for i in range(len(labels)):
            img = images[i].view(1,1,28,28).to(dev)

            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps).to("cpu")
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1 
    writer.add_scalar('accuracy',(correct_count/all_count),num_aver)
    aver_accur = aver_accur + (correct_count/all_count)/10
writer.add_scalar('aver_accur',aver_accur,1)
