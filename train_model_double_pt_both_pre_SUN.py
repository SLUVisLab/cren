from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from random import sample, random
import time
import os
import copy
from RandomErase import RandomErase
from PIL import Image
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir is where your dataset is, store_dir is where to sotre your trained model and anything about this.
data_dir = "/lab/vislab/DATA/SUN397/datasets_SUN"
store_dir= 'train_double_SUN'
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "double"

# Number of classes in the dataset, remember to change to the correct number.
num_classes = 397

# Batch size for training (change depending on how much memory you have)
batch_size = 50

# Number of epochs to train for
num_epochs = 25



# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

class ModelParallelResNet50(nn.Module):
    def __init__(self):

        super(ModelParallelResNet50, self).__init__()

        mod1=models.resnet50(pretrained=True)
        mod2=models.resnet50(pretrained=True)
        self.model1 = nn.Sequential(
            mod1.conv1,
            mod1.bn1,
            mod1.relu,
            mod1.maxpool,

            mod1.layer1,
            mod1.layer2,
            mod1.layer3,
            mod1.layer4,
        ).to("cuda:0")
        self.model2 = nn.Sequential(
            mod2.conv1,
            mod2.bn1,
            mod2.relu,
            mod2.maxpool,

            mod2.layer1,
            mod2.layer2,
            mod2.layer3,
            mod2.layer4,
        ).to("cuda:0")
        self.fc=nn.Linear(4096,num_classes).to('cuda:0')
    def forward(self, x):
        x=x.transpose(1,0)
        x0=x[:-1].transpose(1,0)
        x1=x[-1]
        bs, ncrops, c, h, w = x0.size()
        x0=x0.contiguous().view((-1, c, h, w))
        x0 = self.model1(x0.to('cuda:0'))
        x0 = F.avg_pool2d(x0, 8)
        x0,_ = torch.max(x0.view(bs, ncrops, -1),1)
        x0=x0.to('cuda:0')
        x1= self.model2(x1.to('cuda:0'))
        x1 = F.avg_pool2d(x1, 8)
        x1=x1.view(bs,-1).to('cuda:0')
        x=torch.cat([x0,x1],1)
        if self.training==True:
            x=F.dropout(x,0.4)
        return self.fc(x.view(x.size(0), -1))

def random_crops(img, k):
    crops=[]
    five256=torchvision.transforms.RandomCrop(256)
    for j in range(k):
        im = five256(img)
        crops.append(im)
    return crops
def crops_and_random(img):
    im=np.array(img)
    w, h, _ = im.shape
    if w<h:
        wi=512
        hi=int(wi*h*1.0/w)
    else:
        hi=512
        wi = int(hi * w * 1.0 / h)

    res=torchvision.transforms.Resize((wi,hi),interpolation=2)
    img=res(img)
    Rand=torchvision.transforms.RandomCrop(512)
    re=torchvision.transforms.Resize((256,256),interpolation=2)
    return random_crops(img, 4)+[re(Rand(img))]


def val_crops(img):
    im = np.array(img)
    w, h, _ = im.shape
    if w < h:
        wi = 512
        hi = int(wi * h * 1.0 / w)
    else:
        hi = 512
        wi = int(hi * w * 1.0 / h)
    res=torchvision.transforms.Resize((wi,hi),interpolation=2)
    img=res(img)
    im=np.array(img)
    Rand = torchvision.transforms.RandomCrop(512)
    re = torchvision.transforms.Resize((256, 256), interpolation=2)
    a=int(wi/256)
    b=int(hi/256)
    crs=[]
    for i in range(a):
        for j in range(b):
            crs.append(Image.fromarray((im[i*256:((i+1)*256),j*256:((j+1)*256)]).astype('uint8')).convert('RGB'))
    return sample(crs,4)+[re(Rand(img))]

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []

    best_model_wts_max = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]
    for epoch in range(num_epochs):
        time_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        num_class=np.zeros(num_classes)
        num_wrong=np.zeros(num_classes)
        wrong=np.zeros(num_classes)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #bs, ncrops, c, h, w = inputs.size()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        #outputs = model(inputs.view((-1, c, h, w)))
                        #outputs, _ = torch.max(outputs.view(bs, ncrops, -1), 1)
                        outputs=model(inputs).to(device)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)



                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics

                if phase=='val':

                    for i in range(len(labels.data)):
                        num_class[int(labels.data[i])]+=1

                    for i in range(len(preds)):
                        if preds[i]!=labels.data[i]:
                            num_wrong[int(labels.data[i])]+=1


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase=='train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase=='train':
                train_acc.append(epoch_acc)
            else:
                val_acc.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        time_end = time.time()
        print('time cost', time_end - time_start, 's')
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    all_loss={'train':train_loss,'val':val_loss}
    all_acc={'train':train_acc,'val':val_acc}
    return model, all_loss, all_acc

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_name == "double":
        model_ft = ModelParallelResNet50()
        input_size=256
        ct=0
        for child in model_ft.model1.children():
            ct += 1
            #change the "8" to "7" or "6" to unfreeze one more layer or two
            if ct < 8:
                for param in child.parameters():
                    param.requires_grad = False
        ct=0
        for child in model_ft.model2.children():
            ct += 1
            #change the "8" to "7" or "6" to unfreeze one more layer or two, both need to be changed.
            if ct < 8:
                for param in child.parameters():
                    param.requires_grad = False
    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        ct = 0
        for child in model_ft.children():
            ct += 1
            if ct < 8:
                for param in child.parameters():
                    param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 512

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = 512

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 512

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes

        input_size = 512

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 512

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets

# Detect if we have a GPU available
device = torch.device("cuda:0")

# Send the model to GPU


# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
data_transforms = {
    'train': transforms.Compose([

        transforms.Lambda(lambda img: RandomErase(img)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Lambda(lambda img: crops_and_random(img)),
        #transforms.Resize((512,512),interpolation=2),
        #transforms.Lambda(lambda img: four_and_random(img)),
        transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean=[n / 255.
                                                                                                    for n in
                                                                                                    [129.3, 124.1,
                                                                                                     112.4]],
                                                                                              std=[n / 255. for n in
                                                                                                   [68.2, 65.4,
                                                                                                    70.4]])])(crop) for
                                                     crop in crops]))

    ]),


    'val': transforms.Compose([
        #transforms.Resize((512,512),interpolation=2),
        #transforms.FiveCrop(256),
        transforms.Lambda(lambda img: val_crops(img)),
        transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean=[n / 255.
                                                                                                    for n in
                                                                                                    [129.3, 124.1,
                                                                                                     112.4]],
                                                                                              std=[n / 255. for n in
                                                                                                   [68.2, 65.4,
                                                                                                    70.4]])])(crop) for
                                                     crop in crops]))

    ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

classe=image_datasets['train'].classes
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Observe that all parameters are being optimized


if not os.path.exists(store_dir):
    os.mkdir(store_dir)
# Initialize the model for this run
torch.save(classe,store_dir+'/classes.pth')
torch.save(image_datasets['train'].imgs, store_dir+'/tradsets.pth')
torch.save(image_datasets['val'].imgs, store_dir+'/valdsets.pth')
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# Train and evaluate
model_ft, loss, acc= train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
torch.save(model_ft,store_dir+'/model.pth')
torch.save(loss, store_dir+'/loss.pth')
torch.save(acc, store_dir+'/acc.pth')
f=plt.figure()
plt.plot(range(num_epochs),loss['train'],label='traloss')
plt.plot(range(num_epochs),loss['val'],label='valloss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()
f.savefig(store_dir+'/loss.png',dpi=f.dpi)
g=plt.figure()
plt.plot(range(num_epochs),acc['train'],label='traacc')
plt.plot(range(num_epochs),acc['val'],label='valacc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('acc')
plt.legend()
g.savefig(store_dir+'/acc.png',dpi=g.dpi)
