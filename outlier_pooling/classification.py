from data.datasets import DefaultDataset
from data.transforms import get_transforms
from models import ResNet50
from evaluation import classification_accuracy

from torchvision import transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import torch
import neptune

neptune.init('SLUVisLab/OutlierPooling')

PARAMS = {
    'input_size': 256,
    #'group_size': 8,
    'batch_size': 64,
    #'margin': 0.2,
    'learning_rate': 0.015,
    'embedding_dim': 256,
    'parallel': False,
    'horizontal_flip': 0.25,
    'vertical_flip': 0.25,
    'epochs': 25,
    'pooling_type': "avg",
    'momentum': 0.5,
    'step_size': 5,
    'dataset': "MIT_Indoor"
}
norm_mean = [n / 255 for n in [75.58, 96.37, 92.88]]
norm_std = [n / 255 for n in [43.36, 53.14, 52.06]]
neptune.create_experiment(
    '', 
    params=PARAMS, 
    tags=['classification', '{} pooling'.format(PARAMS['pooling_type']), PARAMS['dataset']], 
    upload_source_files=["*"]
)

data_transforms = {
    'train': {
        'horizontal_flip': [PARAMS['horizontal_flip']],
        'vertical_flip': [PARAMS['vertical_flip']],
        'resize': [(PARAMS['input_size'],PARAMS['input_size'])],
        'normalize': [[n for n in norm_mean], [n for n in norm_std]]
    },
    'validation': {
        'resize': [(PARAMS['input_size'],PARAMS['input_size'])],
        'normalize': [[n for n in norm_mean], [n for n in norm_std]]
    },
}

trainset = DefaultDataset('/lab/vislab/DATA/{}/images'.format(PARAMS['dataset']),
                          '/lab/vislab/DATA/{}/classification/train.csv'.format(PARAMS['dataset']), get_transforms(data_transforms['train']))
validset = DefaultDataset('/lab/vislab/DATA/{}/images'.format(PARAMS['dataset']),
                        '/lab/vislab/DATA/{}/classification/test.csv'.format(PARAMS['dataset']), get_transforms(data_transforms['validation']))

train_loader = DataLoader(trainset, batch_size=PARAMS['batch_size'], shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(validset, batch_size=PARAMS['batch_size'], num_workers=4, drop_last=True)

device = torch.device("cuda")
model = ResNet50(
    pretrained=True, 
    num_classes=len(set(trainset.targets)),
    max_epochs=PARAMS['epochs'],
    dropout=0.0
).to(device)

if PARAMS['parallel']:
    model = nn.DataParallel(model,[0,1,2])

optimizer = torch.optim.SGD(model.parameters(), lr=PARAMS['learning_rate'], momentum=PARAMS['momentum'])
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=PARAMS['step_size'], gamma=0.5)

def train(model, epoch, pooling='outlier'):
    running_loss = 0.0
    running_accuracy = 0.0
    model.train()
    i = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels, paths = batch

        outputs = model(inputs.to(device), epoch, pooling)
        outputs = torch.sigmoid(outputs)

        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.cpu().item()
        running_accuracy += classification_accuracy(outputs.cpu(), labels)

        i += 1

    return running_loss / i, running_accuracy / i

def validation(model, epoch, pooling='outlier'):
    running_loss = 0.0
    running_accuracy = 0.0
    model.eval()
    i = 0
    
    with torch.no_grad():
        for batch in val_loader:
            optimizer.zero_grad()
            inputs, labels, paths = batch

            outputs = model(inputs.to(device), epoch, pooling)
            outputs = torch.sigmoid(outputs)

            loss = criterion(outputs, labels.to(device))

            running_loss += loss.cpu().item()
            running_accuracy += classification_accuracy(outputs.cpu(), labels)
            i += 1

    return running_loss / i, running_accuracy / i

for epoch in range(PARAMS['epochs']):
    print("Epoch", epoch+1)
    val_loss, val_acc = validation(model, epoch, PARAMS['pooling_type'])
    print("Val Acc: {}\tVal Loss: {}".format(val_acc, val_loss))
    neptune.log_metric('Val. Accuracy', val_acc)
    neptune.log_metric('Val. Loss', val_loss)
    
    train_loss, train_acc = train(model, epoch, PARAMS['pooling_type'])
    print("Train Acc: {}\tTrain Loss: {}".format(train_acc, train_loss))
    neptune.log_metric('Train Accuracy', train_acc)
    neptune.log_metric('Train Loss', train_loss)
    
    scheduler.step()
    

