from __future__ import print_function
from __future__ import division
import numpy as np

from PIL import Image
from random import random
def RandomErase(img, p=0.5, s=(0.06,0.12), r=(0.5,1.5)):
    im=np.array(img)
    w,h,_=im.shape
    S=w*h
    pi=random()
    if pi>p:
        return img
    else:

        Se=S*(random()*(s[1]-s[0])+s[0])
        re=random()*(r[1]-r[0])+r[0]
        He=int(np.sqrt(Se*re))
        We=int(np.sqrt(Se/re))
        if He>=h:
            He=h-1
        if We>=w:
            We=w-1
        xe=int(random()*(w-We))
        ye=int(random()*(h-He))
        im[xe:xe+We,ye:ye+He]=int(random()*255)
        return Image.fromarray(im.astype('uint8')).convert('RGB')