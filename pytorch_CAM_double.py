# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import torch
import pdb
from glob import glob
import os
import random
#remember to change to number of classes
num_classes=397
if not os.path.exists("./CAM_test"):
    os.mkdir("./CAM_test")
CAM_dir="./CAM_test"
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
            x=F.dropout(x,0.3)
        return self.fc(x.view(x.size(0), -1))
def resize(img):
    im = np.array(img)
    w, h, _ = im.shape
    if w < h:
        wi = 512
        hi = int(wi * h * 1.0 / w)
    else:
        hi = 512
        wi = int(hi * w * 1.0 / h)
    res = transforms.Resize((wi, hi), interpolation=2)
    return res(img)
def val_crops(img):
    im = np.array(img)
    w, h, _ = im.shape
    if w < h:
        wi = 512
        hi = int(wi * h * 1.0 / w)
    else:
        hi = 512
        wi = int(hi * w * 1.0 / h)
    res=transforms.Resize((wi,hi),interpolation=2)
    img=res(img)
    im=np.array(img)
    Cenc = transforms.CenterCrop(512)
    re = transforms.Resize((256, 256), interpolation=2)
    a=int(wi/256)
    b=int(hi/256)
    crs=[]
    for i in range(a):
        for j in range(b):
            crs.append(Image.fromarray((im[i*256:((i+1)*256),j*256:((j+1)*256)]).astype('uint8')).convert('RGB'))
    l=list(range(len(crs)))
    random.shuffle(l)
    l=l[:4]
    print(l,a,b)
    return [[crs[l[0]],crs[l[1]],crs[l[2]],crs[l[3]]]+[re(Cenc(img))],[(int(l[0]/b)*256,int(l[0]%b)*256),
                                                                       (int(l[1]/b)*256,int(l[1]%b)*256),
                                                                       (int(l[2]/b)*256,int(l[2]%b)*256),
                                                                       (int(l[3]/b)*256,int(l[3]%b)*256)]]






def returnCAM1(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256,256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        fe = F.avg_pool2d(torch.from_numpy(feature_conv), 8)
        fe=fe.squeeze()
        _,idn=torch.max(fe,axis=0)
        pro_layer = np.zeros((bz, nc, h, w))
        for i in range(nc):
            pro_layer[int(idn[i]),i]=np.zeros((h,w))+1
        feature_conv=feature_conv*pro_layer
        sub_cam = []
        min=10000
        max=-10000
        for i in range(4):
            cam = weight_softmax[idx].dot(feature_conv[i].reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            if min>np.min(cam):
                min=np.min(cam)
            if max<np.max(cam):
                max=np.max(cam)
            sub_cam.append(cam)
        for i in range(4):
            sub_cam[i]=sub_cam[i]-min
            sub_cam[i]=sub_cam[i]/max
            sub_cam[i]=np.uint8(sub_cam[i]*255)
            sub_cam[i]=cv2.resize(sub_cam[i], size_upsample,interpolation=cv2.INTER_NEAREST)
        output_cam.append(sub_cam)
    return output_cam

def returnCAM2(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256,256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


process1 = transforms.Compose([
        transforms.Lambda(lambda img: val_crops(img)),
        transforms.Lambda(lambda crops: [torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean=[n / 255.
                                                                                                    for n in
                                                                                                    [129.3, 124.1,
                                                                                                     112.4]],
                                                                                              std=[n / 255. for n in
                                                                                                   [68.2, 65.4,
                                                                                                    70.4]])])(crop) for crop in crops[0]]),
    crops[1]])
    ])


def hook_feature1(module, input, output):
    features_blobs1.append(output.data.cpu().numpy())

def hook_feature2(module, input, output):
    features_blobs2.append(output.data.cpu().numpy())
# this is where your file comes from 
file="/lab/vislab/DATA/SUN397/datasets_SUN/val/abbey/sun_awmdgbfmljliozsj.png"
img_pil = Image.open(file)
# where your trained model comes from
net = torch.load('train_double_SUN/model.pth')
finalconv_name = "7"
net.eval()
features_blobs1 = []
features_blobs2 = []
img_tensor = process1(img_pil)
net.model1._modules.get(finalconv_name).register_forward_hook(hook_feature1)
net.model2._modules.get(finalconv_name).register_forward_hook(hook_feature2)

img_variable = img_tensor[0].unsqueeze(0)
logit = net(img_variable.cuda())
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()
params = list(net.parameters())
weight_softmax1 = np.squeeze(params[-2].data.cpu().numpy())[:,:2048]
weight_softmax2 = np.squeeze(params[-2].data.cpu().numpy())[:,-2048:]
CAMs1 = returnCAM1(features_blobs1[0], weight_softmax1, [idx[0]])
CAMs2 = returnCAM2(features_blobs2[0], weight_softmax2, [idx[0]])

heatmap2 = cv2.applyColorMap(cv2.resize(CAMs2[0], (512, 512)), cv2.COLORMAP_JET)
img=np.array(resize(img_pil))
w,h,c=img.shape
he1=np.zeros((w,h,c))
print(img_tensor[1])
for i in range(4):
    heatmap1 = cv2.applyColorMap(cv2.resize(CAMs1[0][i], (256, 256)), cv2.COLORMAP_JET)
    he1[img_tensor[1][i][0]:img_tensor[1][i][0]+256,img_tensor[1][i][1]:img_tensor[1][i][1]+256,0:c]=heatmap1
he2=np.zeros((w,h,c))
he2[int((w-512)/2):int((w-512)/2)+512,int((h-512)/2):int((h-512)/2)+512,0:c]=heatmap2
img = cv2.imread(file)
img=cv2.resize(img,(h,w))
print(img.shape)
result1=0.5*img+0.4*he1
result2=0.5*img+0.4*he2
result=0.4*img+0.3*he1+0.3*he2
cv2.imwrite(CAM_dir+"/"+'result.jpg', result)
cv2.imwrite(CAM_dir+"/"+'result1.jpg', result1)
cv2.imwrite(CAM_dir+"/"+'result2.jpg', result2)
cv2.imwrite(CAM_dir+"/"+'source.jpg', img)
print(idx[0])

