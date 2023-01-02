
#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import os
# import smdebug.pytorch as smd
import logging


from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
print(torch.__version__)
logger.info(torch.__version__)
from PIL import ImageFile
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader,criterion, data_namee,hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook.set_mode(smd.modes.EVAL)

    correct = 0
    running_loss=0
    model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss =criterion(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            running_loss +=  test_loss.item()


    running_loss /= len(test_loader.dataset)

#     print("\n"+data_namee+ ": Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
#             test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
#         )
#     )
    logger.info(f"Loss {data_namee} {running_loss}")
    logger.info(f"accuracy {data_namee} {100.0 * correct / len(test_loader.dataset)}")
 




def train(model, train_loader, criterion, optimizer, epoch, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook.set_mode(smd.modes.TRAIN)
    model.train()
    correct = 0
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        running_loss +=  loss.item()
        optimizer.step()
        pred=output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 5 == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t avg_Loss: {:.2f} \t batch_indx: {} / {}, accuracy: {} %".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    running_loss/(args.batch_size*(batch_idx+1)),
                    batch_idx,
                    len(train_loader),
                    100*(correct/(args.batch_size*(batch_idx+1)))
                ))
            
    
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133)
                            )

    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = "/opt/ml/input/data/train/"
    test_data_path = "/opt/ml/input/data/test/"
    validation_data_path= "/opt/ml/input/data/valid/"
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
                                                            
    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        ])

    #Get data from s3
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(),lr=args.lr)
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)        

    hook.register_loss(criterion)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    test(model, train_loader, criterion, "On Training dataset before training", hook)   

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, hook)
        test(model, train_loader, criterion, "On Training dataset after the prev. epoch ", hook)    

        test(model, validation_loader, criterion, "On Validation dataset", hook)    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, "On testing_data:", hook)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    
    
if __name__=='__main__':

        
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        metavar="N",
        help="input batch size for training (default: 50)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for testing (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        '--data', 
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"],
        help="data path (s3://sagemaker-us-east-1-457357529781/data/dogImages/) ")
    
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    args=parser.parse_args()
    
    main(args)
