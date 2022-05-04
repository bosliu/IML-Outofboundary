import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from tqdm import tqdm
import numpy as np
import random
import os
from dataclasses import dataclass
from PIL import Image

import argparse

from typing import Generator, List, Optional

"""
********************  FILE STRUCTURE  ********************
---- $Current directory (path)
   |
   |---- main.py
   |---- dataset
       |
       |---- *.txt (Labels)
       |
       |---- food
            |
            |---- *.jpg (Images)
            |
            |---- processed
                |
                |---- *.jpg (Images, Processed)
   
"""
# hyperparams
IMAGE_DIM = (460, 310)
VALIDATION = True
VALIDATION_PROP = 0.15
TRAIN_LABEL_TRIPLETS_PROP = 1

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 15
TRAIN = True
LOAD_MODEL = False
MODEL_PATH = 'trained_models/'
MODEL_FILE_NAME = 'model_epoch_10.pth'
TEST = False

class Preprocess:
    def __init__(self, img_path, lbl_path, use_preprocessed_img=True) -> None:
        self.lbl_path = lbl_path
        self.to_perform = False
        if use_preprocessed_img:
            processed_img_path = os.path.join(img_path, 'processed')
            self.img_path = processed_img_path
            try:
                assert len(os.listdir(processed_img_path)) == len([f for f in os.listdir(img_path) if f.endswith('.jpg')])
                print("Processed images found at:" + self.img_path)
            except:
                self.to_perform = True
        else:
            self.img_path = img_path
        if self.to_perform:
            if not os.path.exists(processed_img_path):
                os.mkdir(os.path.join(img_path, 'processed'))
            self.img_path_ = os.path.join(img_path, 'processed')
        self.check_img_num()
        self.images = []
        self.train_triplets = []
        self.validation_triplets = None
        self.test_triplets = []

    def jpg_gen(self) -> Generator:
        for f in os.listdir(self.img_path):
            if f.endswith(".jpg"):
                yield f
    
    def check_img_num(self):
        self.img_num = len(list(set(self.jpg_gen())))
        print(f"Total # of img found: {self.img_num}.")
    
    def load_img(self, out=False) -> Optional[List[np.ndarray]]:
        images = []
        state = " Loading images " if not self.to_perform else " Loading and Preprocessing images "
        print("\n---------->>" + state + "<<----------")
        pb = tqdm(total=self.img_num, desc=state, position=0)
        for i, im in enumerate(self.jpg_gen()):
            im = cv.imread(os.path.join(self.img_path, im))
            im = self.img_process(i, im)
            images.append(im)
            pb.update()
        self.images = images
        if out:
            return self.images
        print(f"\nFinished with {len(self.images)} images.")

    def read_labels(self):
        print("\n---------- Reading labels ----------")
        train_triplets = []
        test_triplets = []
        with open(self.lbl_path + '/train_triplets.txt', 'r') as f:
            for l in f:
                a, b, c = tuple(map(int, l.split( )))
                train_triplets.append((a, b, c))
        self.train_triplets = train_triplets
        random.shuffle(self.train_triplets)
        print(f"Train labels read with total length of {len(self.train_triplets)}.")
        with open(self.lbl_path + '/test_triplets.txt', 'r') as f:
            for l in f:
                a, b, c = tuple(map(int, l.split( )))
                test_triplets.append((a, b, c))
        self.test_triplets = test_triplets
        print(f"Test labels read with total length of {len(self.test_triplets)}.")
    
    def split_validation(self):
        if VALIDATION:
            train_len = int(len(self.train_triplets) * (1- VALIDATION_PROP))
            self.train_triplets, self.validation_triplets = self.train_triplets[:train_len], self.train_triplets[train_len:]
        else:
            pass
    
    def img_process(self, i, img):
        # image process here...
        assert img is not None
        if self.to_perform:
            img = cv.resize(img, dsize=IMAGE_DIM, interpolation=cv.INTER_LANCZOS4)
            img = cv.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21)
            cv.imwrite(os.path.join(self.img_path_, '{:05d}.jpg'.format(i)), img)
        return img

    def prep_pipeline(self):
        self.load_img()
        self.read_labels()
        self.split_validation()
        return self.images, self.train_triplets, self.validation_triplets, self.test_triplets


class FoodDataSet(Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.tf = transform if transform != None else T.Compose(
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            )
    
    def __getitem__(self, index):
        a, b, c = self.triplets[index]
        ims = [Image.fromarray(cv.cvtColor(images[i], cv.COLOR_BGR2RGB)) for i in (a, b, c)]
        return tuple([self.tf(im) for im in ims])
    
    def __len__(self):
        return len(self.triplets)


class NeuralNet(nn.Module):
    EMBEDDING_DIM = 64
    def __init__(self, inp: int = 2208, hidden: int = 384, hidden_2: int = 128, d: float = 0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inp, hidden),
            nn.PReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(d),
            nn.Linear(hidden, hidden_2),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_2),
            nn.Dropout(d),
            nn.Linear(hidden_2, self.EMBEDDING_DIM)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)


def closest_image(a, b, c):
    """
    Returns: 
        int: 1 if a is more similar to the positive image b. 
                0 if a is more similar to the negative image c.
                
    Note:
        Works for batches. 
        Each input tensor should have this shape: `N x W x H x C` 
        where `N` is the number of samples.
    """
    pairwise_dist = nn.PairwiseDistance(p=2)
    distance_pos = pairwise_dist(a, b)
    distance_neg = pairwise_dist(a, c)
    return (~(distance_pos >= distance_neg).to(torch.bool)).to(torch.int8)


def accuracy(pred, gt):
    return np.mean((pred == gt))


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    assert model_name in ("densenet", "resnet", "inception"), ValueError("Wrong model name!")
    if model_name == "densenet":
        """ Densenet
        """
        model_ft = torchvision.models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = NeuralNet(inp=num_ftrs) 
        input_size = 224
    elif model_name == "resnet":
        """ Resnet152
        """
        model_ft = torchvision.models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = NeuralNet(inp=num_ftrs) 
        input_size = 224
    elif model_name == "inception":
        """ Inception v3
        Expects (299,299) sized images and has auxiliary output
        """
        model_ft = torchvision.models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, NeuralNet.EMBEDDING_DIM)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = NeuralNet(inp=num_ftrs)
        input_size = 299

    return model_ft, input_size


@dataclass
class AverageMeter:
    name: str = ""
    fmt: str = ':f'
    val: float = 0
    avg: float = 0
    sum: float = 0
    count: int = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.val:.3f}, avg: {self.avg:.3f}.'


def train(model, optimizer, train_loader, criterion, batch_size, epochs, 
          model_name='model', is_inception=False, val_loader=None):
    print(len(train_loader))
    batch_num = len(train_loader)
    pbar = tqdm(desc='Training', total=epochs * batch_num, leave=False)
    
    model_path = os.path.join(path, MODEL_PATH)
    if not os.path.exists(model_path):
        print(".... Making dir at " + str(model_path))
        os.mkdir(model_path)
    
    if val_loader:
        val_acc = 0
    for epoch in range(epochs):
        print("\n>>>> Training")
        pbar.write(f"Epoch {epoch + 1} / {epochs}")
        model.train()
        losses = AverageMeter('Loss', ':.3f')
        for batch_idx, (ima, imb, imc) in enumerate(train_loader):
            ima = ima.to(device)
            imb = imb.to(device)
            imc = imc.to(device)
            pbar.set_description_str(f'Epoch {epoch + 1}, batch {batch_idx}')
            optimizer.zero_grad()
            
            if is_inception:
                raise NotImplementedError
            else:
                out = tuple(map(model, [ima, imb, imc]))
                loss = criterion(*out)
            losses.update(loss.item(), ima.shape[0])
            loss.backward()
            optimizer.step()
            
            pred_labels = closest_image(*out).detach().cpu().numpy()
            acc = accuracy(pred_labels, np.ones_like(pred_labels))
            pbar.set_postfix_str("{}, accuracy {:.3f}".format(losses, acc))
            pbar.update()
            # if batch_idx > 3:
            #     break

            # do validation here to follow the training process
        if epoch % 2 == 0 and val_loader:
            print("\n>>>> Validation")
            model.eval()
            with torch.no_grad():
                correct = 0
                best_epoch = -1
                for (ima, imb, imc) in val_loader:
                    ima = ima.to(device)
                    imb = imb.to(device)
                    imc = imc.to(device)
                    im = (ima, imb, imc)
                    out = tuple(map(model, im))
                    pred_labels = closest_image(*out).detach().cpu().numpy()
                    correct += np.sum((pred_labels == np.ones_like(pred_labels)).astype(np.uint8))
                curr_acc = correct / len(validation_set)
                print(f"\nCurrent accuracy for the validation data is {curr_acc:.3f} with.")
                if val_acc < curr_acc:
                    val_acc = curr_acc
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_path, model_name + f'_better_valacc_at_epoch_{epoch+1}.pth'))
        torch.save(model.state_dict(), os.path.join(model_path, model_name + f'_epoch_{epoch+1}.pth'))
        pbar.refresh()
    print("\n---------->> Training Results <<----------")
    print(f"Epochs finished {epoch + 1} / {epochs}.")
    if val_loader:
        print(f"Best validation accuracy achieved at {best_epoch} with {curr_acc:.3f}.")
    print("Models are saved at " + model_path)
    

def predict(model, test_loader):
    print("\n>>>> Prediction")
    """Predict 0/1 for the given predict_set (Dataset)"""
    choices = np.array([])
    # List of tuples which the model chose as most similar
    model.eval()
    with torch.no_grad():
        for (ima, imb, imc) in tqdm(test_loader, leave=True, total= len(test_loader)):
            # Get embeddings from our model
            ima = ima.to(device)
            imb = imb.to(device)
            imc = imc.to(device)
            im = (ima, imb, imc)
            out = tuple(map(model, im))
            # Compute distances and the corresponding labels
            labels = closest_image(*out).detach().cpu().numpy()

            choices = np.concatenate((choices, labels.reshape(-1))).reshape(-1)
    # result = np.array(choices).flatten()
    # return np.concatenate(result).ravel()
    return choices.astype(np.uint8).tolist()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--no_validation', action="store_true")
    parser.add_argument('--validation_prop', type=float, required=False, default=VALIDATION_PROP)
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--batch_size', type=int, required=False, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, required=False, default=LR)
    parser.add_argument('--epochs', type=int, required=False, default=EPOCHS)
    args = parser.parse_args()
    
    TRAIN = args.train
    VALIDATION = not args.no_validation
    assert 0 < args.validation_prop and args.validation_prop < 1, ValueError("Wrong validation proportion!")
    VALIDATION_PROP = args.validation_prop
    LOAD_MODEL = args.load
    TEST = args.test
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    global path
    path = os.getcwd()
    if 'dataset' not in os.listdir():
        path = os.path.join(path, 'task3')  # adaption to path
    path_label = os.path.join(path, 'dataset')
    path_img = os.path.join(path_label, 'food')
    print("Current path: " + path)
    print("Label path: " + path_label)
    print("Image path: " + path_img)
    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device {}.".format(device))
    torch.manual_seed(2022)
    np.random.seed(2022)
    
    # preprocess on data
    prep = Preprocess(img_path=path_img, lbl_path=path_label, use_preprocessed_img=True)
    global images, train_triplets, validation_triplets, test_triplets
    images, train_triplets, validation_triplets, test_triplets = prep.prep_pipeline()
    
    # Initialize the pre-trained network
    feature_extract = True
    model_type = 'densenet'
    model_pretrained, input_size = initialize_model(model_type, feature_extract, use_pretrained=True)
    print("\n---------- Pre-trained Model ----------")
    # print(model_pretrained)
    
    model_pretrained = model_pretrained.to(device)
    params_to_update = model_pretrained.parameters()
    # print("Params to learn:")

    if feature_extract:
        params_to_update = []
        for name,param in model_pretrained.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                # print("\t",name)
    else:
        # for name,param in model_pretrained.named_parameters():
        #     if param.requires_grad:
        #         print("\t",name)
        pass

    # Then the dataset and dataloader init, together with transform
    train_tf = T.Compose([
        T.RandomSizedCrop(input_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    validation_tf = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if TRAIN:
        train_set = FoodDataSet(train_triplets, transform=train_tf)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        
    valid_loader = None
    if VALIDATION:
        validation_tf = train_tf
        validation_set = FoodDataSet(validation_triplets, transform=validation_tf)
        valid_loader = DataLoader(validation_set, batch_size=BATCH_SIZE*2, shuffle=False, drop_last=False)
        
    if TEST:
        test_tf = validation_tf
        test_set = FoodDataSet(test_triplets, transform=test_tf)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE*2, shuffle=False, drop_last=False)
    
    optimizer = torch.optim.Adam(params_to_update, lr=LR)
    criterion = nn.TripletMarginLoss(margin=0.5)
    
    if TRAIN:
        train(model=model_pretrained, optimizer=optimizer, train_loader=train_loader, criterion=criterion, 
            batch_size=BATCH_SIZE, epochs=EPOCHS, model_name='model', is_inception=False, val_loader=valid_loader)
    
    if LOAD_MODEL:
        try:
            model_pretrained.load_state_dict(torch.load(os.path.join(path, MODEL_PATH + MODEL_FILE_NAME)))
        except:
            ValueError("Wrong load configuration!")
    
    if TEST:
        results = predict(model=model_pretrained, test_loader=test_loader)
        with open(os.path.join(path, 'prediction.txt'), 'w') as f:
            for r in results:
                s = ''.join(str(r))
                f.write(s + '\n')
        print("Output completed as " + str(os.path.join(path, 'prediction.txt')))
