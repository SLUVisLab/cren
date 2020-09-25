

from PIL import Image
from torchvision import models, transforms, datasets
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import torch
from glob import glob
import os
import random
import json
from pytorch_lightning.core.lightning import LightningModule
num_classes=100
CAM_dir="./CAM_test"
if not os.path.exists(CAM_dir):
    os.mkdir(CAM_dir)
data_dir="/lab/vislab/OPEN/datasets_RGB_one"
"""
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
"""
class Model(LightningModule):
    """ Model
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.epoch = 0
        self.learning_rate = 0.015
        self.loss = nn.CrossEntropyLoss()
        self.training_correct_counter = 0
        self.training = False

        mod1 = models.resnet50(pretrained=True)
        mod2 = models.resnet50(pretrained=True)
        self.model1 = nn.Sequential(
            mod1.conv1,
            mod1.bn1,
            mod1.relu,
            mod1.maxpool,

            mod1.layer1,
            mod1.layer2,
            mod1.layer3,
            mod1.layer4,
        )
        self.model2 = nn.Sequential(
            mod2.conv1,
            mod2.bn1,
            mod2.relu,
            mod2.maxpool,

            mod2.layer1,
            mod2.layer2,
            mod2.layer3,
            mod2.layer4,
        )
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = x.transpose(1, 0)
        x0 = x[:-1].transpose(1, 0)
        x1 = x[-1]
        bs, ncrops, c, h, w = x0.size()
        x0 = x0.contiguous().view((-1, c, h, w))
        x0 = self.model1(x0)
        x0 = F.avg_pool2d(x0, 8)
        x0, _ = torch.max(x0.view(bs, ncrops, -1), 1)
        x1 = self.model2(x1)
        x1 = F.avg_pool2d(x1, 8)
        x1 = x1.view(bs, -1)
        x = torch.cat([x0, x1], 1)
        if self.training == True:
            x = F.dropout(x, 0.4)
        return self.fc(x.view(x.size(0), -1))
    def eval(self):
        self.model1.eval()
        self.model2.eval()
        self.fc.eval()
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
        output_cam.append(cv2.resize(cam_img, size_upsample,interpolation=cv2.INTER_NEAREST))
    return output_cam


process1 = transforms.Compose([
        transforms.Lambda(lambda img: val_crops(img)),
        transforms.Lambda(lambda crops: [torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean=[n / 255.
                                                                                                    for n in
                                                                                                    [75.58, 96.37, 92.88]],
                                                                                              std=[n / 255. for n in
                                                                                                   [43.36, 53.14, 52.06]])])(crop) for crop in crops[0]]),
    crops[1]])
    ])
image_datasets = datasets.ImageFolder(os.path.join(data_dir, "val"), process1)

classe=image_datasets.classes
#file="/www/student/cren2/public_html/TERRA/datasets_RGB_one/val/PI_22913/2017-06-02__14-01-34-608.png"

def CAM_double(file,net):
    def hook_feature1(module, input, output):
        features_blobs1.append(output.data.cpu().numpy())

    def hook_feature2(module, input, output):
        features_blobs2.append(output.data.cpu().numpy())

    img_pil = Image.open(file)

    finalconv_name = "7"
    net.eval()
    features_blobs1 = []
    features_blobs2 = []
    img_tensor = process1(img_pil)
    net.model1._modules.get(finalconv_name).register_forward_hook(hook_feature1)
    net.model2._modules.get(finalconv_name).register_forward_hook(hook_feature2)

    img_variable = img_tensor[0].unsqueeze(0)
    net=net.cuda()
    logit = net(img_variable.cuda())
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
    params = list(net.parameters())
    weight_softmax1 = np.squeeze(params[-2].data.cpu().numpy())[:, :2048]
    weight_softmax2 = np.squeeze(params[-2].data.cpu().numpy())[:, -2048:]
    CAMs1 = returnCAM1(features_blobs1[0], weight_softmax1, [idx[0]])
    CAMs2 = returnCAM2(features_blobs2[0], weight_softmax2, [idx[0]])
    heatmap2 = cv2.applyColorMap(cv2.resize(CAMs2[0], (512, 512)), cv2.COLORMAP_JET)
    img = np.array(resize(img_pil))
    w, h, c = img.shape
    he1 = np.zeros((w, h, c))
    for i in range(4):
        heatmap1 = cv2.applyColorMap(cv2.resize(CAMs1[0][i], (256, 256)), cv2.COLORMAP_JET)
        he1[img_tensor[1][i][0]:img_tensor[1][i][0] + 256, img_tensor[1][i][1]:img_tensor[1][i][1] + 256,
        0:c] = heatmap1
    he2 = np.zeros((w, h, c))
    he2[int((w - 512) / 2):int((w - 512) / 2) + 512, int((h - 512) / 2):int((h - 512) / 2) + 512, 0:c] = heatmap2
    img = cv2.imread(file)
    img = cv2.resize(img, (h, w))
    result1 = 0.5 * img + 0.4 * he1
    result2 = 0.5 * img + 0.4 * he2
    result = 0.4 * img + 0.3 * he1 + 0.3 * he2
    fi = file.split("/")
    print(classe[idx[0]],fi[-2])
    return probs[0]-probs[1], classe[idx[0]]==fi[-2], result,result1,result2,img
net = Model.load_from_checkpoint("/www/student/cren2/public_html/TERRA/lightning_one/default/version_7/checkpoints/epoch=12.ckpt")

for cla in classe[:3]:
    files = random.sample(glob("/lab/vislab/OPEN/datasets_RGB_one/val/" + cla + "/*.png"), 40)
    if not os.path.exists(CAM_dir + "/" + cla + "/" + "wrong"):
        os.makedirs(CAM_dir + "/" + cla + "/" + "wrong")
    if not os.path.exists(CAM_dir + "/" + cla + "/" + "right"):
        os.makedirs(CAM_dir + "/" + cla + "/" + "right")
    d = {}

    for file in files:
        fi = file.split("/")
        f = fi[-1][:-4]
        p, i, result, result1, result2, img = CAM_double(file, net)
        print(i)
        if i:
            if not os.path.exists(CAM_dir + "/" + cla + "/" + "right/" + f):
                os.makedirs(CAM_dir + "/" + cla + "/" + "right/" + f)
            cv2.imwrite(CAM_dir + "/" + cla + "/" + "right/" + f + "/" + 'result.jpg', result)
            cv2.imwrite(CAM_dir + "/" + cla + "/" + "right/" + f + "/" + 'result1.jpg', result1)
            cv2.imwrite(CAM_dir + "/" + cla + "/" + "right/" + f + "/" + 'result2.jpg', result2)
            cv2.imwrite(CAM_dir + "/" + cla + "/" + "right/" + f + "/" + 'source.jpg', img)
            d[f] = p
        else:
            if not os.path.exists(CAM_dir + "/" + cla + "/" + "wrong/" + f):
                os.makedirs(CAM_dir + "/" + cla + "/" + "wrong/" + f)
            cv2.imwrite(CAM_dir + "/" + cla + "/" + "wrong/" + f + "/" + 'result.jpg', result)
            cv2.imwrite(CAM_dir + "/" + cla + "/" + "wrong/" + f + "/" + 'result1.jpg', result1)
            cv2.imwrite(CAM_dir + "/" + cla + "/" + "wrong/" + f + "/" + 'result2.jpg', result2)
            cv2.imwrite(CAM_dir + "/" + cla + "/" + "wrong/" + f + "/" + 'source.jpg', img)
    s = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    s = {k: s[k] for k in list(s)[:10]}
    print(s)
    with open(CAM_dir + "/" + cla+"/"+'file.txt', 'w') as file:
        print(s,file=file)