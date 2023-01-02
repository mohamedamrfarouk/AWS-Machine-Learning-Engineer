# i faced a problem to use deployed endpoint so this was the solution
#https://knowledge.udacity.com/questions/841695
import json, logging, sys, os, io, requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JPEG_CONTENT_TYPE = 'image/jpeg'
JSON_CONTENT_TYPE = 'application/json'
ACCEPTED_CONTENT_TYPE = [ 'image/jpeg' ] #Add support for jpeg images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def net():

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model


def model_fn(model_dir):
    logger.info("Into model_fn")
    logger.info(f"Model dir: {model_dir}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")
    model = net().to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("Model loading...")
        model.load_state_dict(torch.load(f, map_location = device))
        logger.info("Model loaded")
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):

    logger.info('Into input_fn.')
    # Process an image uploaded to the endpoint
    logger.info(f'Content-Type: {content_type}')
    logger.info(f'Request-Body-Type: {type(request_body)}')
    if content_type in ACCEPTED_CONTENT_TYPE:
        logger.info(f"Return-Image-Type {content_type}" )
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception(f"Unsupported Content-Type: {content_type}, Accepted Content-Type: {ACCEPTED_CONTENT_TYPE}")

        
def predict_fn(input_object, model):   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Predicting...")
    test_transform =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() ]
    )
    logger.info("Transforms to the input")
    input_object=test_transform(input_object)
    if torch.cuda.is_available():
        input_object = input_object.cuda() #put data into GPU
    logger.info("Transforms ready")
    model.eval()
    with torch.no_grad():
        logger.info("Calling model...")
        prediction = model(input_object.unsqueeze(0))
    return prediction