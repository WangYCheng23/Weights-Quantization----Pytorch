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
    writer = SummaryWriter('runs/dissertation/Test4b/{}'.format(step))
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

    criterion = nn.NLLLoss().to(dev) 
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)

    time0 = time()
    Iterations = 5
    epochs = 10
    running_loss_list= []
    epochs_list = []
    bit = 3

    def quantization(weights,bit,boundry_value):
        param = (1/boundry_value)*(2**bit)
        q_weights = (torch.trunc(param*weights))/param
        q_weights = q_weights.clamp(min=-boundry_value,max=boundry_value)
        return q_weights

    def classify(img, ps):
        ''' 
        Function for viewing an image and it's predicted classes.
        '''
        img = img.to("cpu")
        ps = ps.data.numpy().squeeze()

        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        ax1.imshow(img.numpy().squeeze())
        ax1.axis('off')
        ax2.barh(np.arange(10), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(np.arange(10))
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()

    # for index,val in enumerate(sequence):
    for iterations in range(Iterations):
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
                print("Epoch {} - Training loss: {}".format((iterations*epochs)+e, running_loss/len(trainloader)))
                writer.add_scalar('Loss',running_loss,(iterations*epochs)+e)
                # Document Weights
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

                # Testing
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

                #print("Number Of Images Tested =", all_count)
                writer.add_scalar('Accuracy',(correct_count/all_count),(iterations*epochs)+e)
                print("Model Accuracy =", (correct_count/all_count))


        if iterations == 0:
            model.c1.c1.c1.weight.grad = None
            model.c1.c1.c1.weight.requires_grad = False
            c1_weights = model.c1.c1.c1.weight
            q_c1_weights = quantization(c1_weights,bit=bit,boundry_value=1)#(torch.trunc((2**bit)*c1_weights))/(2**bit)
            model.state_dict()['c1.c1.c1.weight'].data.copy_(q_c1_weights)

        elif iterations == 1:
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

        elif iterations == 2:
            model.c3.c3.c3.weight.grad = None
            model.c3.c3.c3.weight.requires_grad = False
            c3_weights = model.c3.c3.c3.weight
            q_c3_weights = quantization(c3_weights,bit=bit,boundry_value=0.1)#(torch.trunc((2**bit)*c3_weights))/(2**bit)
            model.state_dict()['c3.c3.c3.weight'].data.copy_(q_c3_weights)

        elif iterations == 3:
            model.f4.f4.f4.weight.grad = None
            model.f4.f4.f4.weight.requires_grad = False
            f4_weights = model.f4.f4.f4.weight
            q_f4_weights = quantization(f4_weights,bit=bit,boundry_value=0.5)#(torch.trunc((2**bit)*f4_weights))/(2**bit)
            model.state_dict()['f4.f4.f4.weight'].data.copy_(q_f4_weights)

    print("\nTraining Time (in minutes) =",(time()-time0)/60)