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

for step in range(5):
    writer = SummaryWriter('runs/dissertation/optim/4b_layer4/{}'.format(step))
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    time0 = time()
    Num_q_layer = 5
    epochs = 20
    running_loss_list= []
    epochs_list = []
    bit = 3
    sequence = [1,0.8,0.5,0.4,0.2,0.1,0.08,0.05,0.04,0.02,0.01]

    def quantization(weights,bit,boundry_value):
        param = (1/boundry_value)*(2**bit)
        q_weights = (torch.round(param*weights))/param
        q_weights = q_weights.clamp(min=-boundry_value,max=boundry_value)
        return q_weights

    for index,val in enumerate(sequence):

        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

        trainset = datasets.MNIST(r'..\input\MNIST', download=True, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

        testset = datasets.MNIST(r'..\input\MNIST', download=True, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)

        model=LeNet5()
        model.to(dev)

        criterion = nn.NLLLoss().to(dev) 
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)

        for num_q_layer in range(Num_q_layer):
            # Training
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
                    # This is where the model learns by backpropagating
                    loss.backward()          
                    # And optimizes its weights here
                    optimizer.step()
                    running_loss += loss.item()
                else:
                    print("Step{} - Epoch {} - Training loss: {}".format(step,(num_q_layer*epochs)+e, running_loss/len(trainloader)))
                    # writer.add_scalar('Loss',running_loss,(num_q_layer*epochs)+e)

            if num_q_layer == 1:
                model.c1.c1.c1.weight.grad = None
                model.c1.c1.c1.weight.requires_grad = False
                c1_weights = model.c1.c1.c1.weight
                q_c1_weights = quantization(c1_weights,bit=bit,boundry_value=1)#(torch.trunc((2**bit)*c1_weights))/(2**bit)
                model.state_dict()['c1.c1.c1.weight'].data.copy_(q_c1_weights)

            elif num_q_layer == 2:
                model.c2_1.c2.c2.weight.grad = None
                model.c2_2.c2.c2.weight.grad = None
                model.c2_1.c2.c2.weight.requires_grad = False
                model.c2_2.c2.c2.weight.requires_grad = False
                c2_1_weights = model.c2_1.c2.c2.weight
                c2_2_weights = model.c2_2.c2.c2.weight
                q_c2_1_weights = quantization(c2_1_weights,bit=bit,boundry_value=0.4)#(torch.trunc((2**bit)*c2_1_weights))/(2**bit)
                q_c2_2_weights = quantization(c2_2_weights,bit=bit,boundry_value=0.4)#(torch.trunc((2**bit)*c2_2_weights))/(2**bit)
                model.state_dict()['c2_1.c2.c2.weight'].data.copy_(q_c2_1_weights)
                model.state_dict()['c2_2.c2.c2.weight'].data.copy_(q_c2_2_weights)

            elif num_q_layer == 3:
                model.c3.c3.c3.weight.grad = None
                model.c3.c3.c3.weight.requires_grad = False
                c3_weights = model.c3.c3.c3.weight
                q_c3_weights = quantization(c3_weights,bit=bit,boundry_value=0.1)#(torch.trunc((2**bit)*c3_weights))/(2**bit)
                model.state_dict()['c3.c3.c3.weight'].data.copy_(q_c3_weights)

            elif num_q_layer == 4:
                model.f4.f4.f4.weight.grad = None
                model.f4.f4.f4.weight.requires_grad = False
                f4_weights = model.f4.f4.f4.weight
                q_f4_weights = quantization(f4_weights,bit=bit,boundry_value=val)#(torch.trunc((2**bit)*f4_weights))/(2**bit)
                model.state_dict()['f4.f4.f4.weight'].data.copy_(q_f4_weights)
        else:
            aver_accur = 0.0
            for num_aver in range(4):
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
                aver_accur = aver_accur + (correct_count/all_count)/4
            writer.add_scalar('Accuracy_scaling_val',aver_accur,index)

    print("\nTraining Time (in minutes) =",(time()-time0)/60)