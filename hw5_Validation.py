#%matplotlib inline
import argparse
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as tvt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import cv2
import torchvision.models as models
import torch.nn.functional as F
import urllib
import torch.optim as optim
parser = argparse.ArgumentParser ( description = 'HW04 COCO downloader')
parser.add_argument ( '--root_path' , required = True , type =str )
parser.add_argument ( '--coco_json_path_train', required = True ,type = str )
parser.add_argument ( '--coco_json_path_val', required = True ,type = str )
parser.add_argument ( '--class_list' , required = True , nargs ='*' , type = str )
parser.add_argument ( '--images_per_class' , required = True ,type = int )
args , args_other = parser.parse_known_args ()
class coco_downloader():
    def __init__(self,x,y,z,w,tv,transform):
        #pylab.rcParams['figure.figsize'] = (8.0, 10.0)
        annFile=w
        coco=COCO(annFile)
        images_per_class=z
        from PIL import Image, ImageDraw, ImageFont
        import PIL
        self.img_dict={}
        self.im={}
        self.class_list_index = 0
        dtype = torch.float64
        for id in y:
         catIds = coco.getCatIds(catNms=[id]);
         imgIds = coco.getImgIds(catIds=catIds);
         #print("imgIds", imgIds)
         cats_ann_id = (coco.getAnnIds(catIds=catIds))
         cats_ann = coco.loadAnns(cats_ann_id)
         self.bbox_num_img_cat = np.asarray([cat['bbox'] for cat in cats_ann])
         imgid = [cat['image_id'] for cat in cats_ann]

         #print(bbox_num_img_cat)
         self.demo_img=[]
         for i in range(images_per_class):
             if not os.path.exists(x):
                os.makedirs((x))
             if not os.path.exists(os.path.join(x,tv)):
                os.makedirs(os.path.join(x,tv))
             img_path=os.path.join(x, tv, id)
             if not os.path.exists(img_path):
               os.mkdir(img_path, mode=0o666)
             img = coco.loadImgs(imgid[i])[0]
             try:
               image1 = io.imread(img['coco_url'])
             except:
                 print("fail")
                 return "fail"

             img = Image.fromarray(image1)
             draw = ImageDraw.Draw(img)

             img= img.resize((64, 64), Image.BOX)

             img.save(img_path+"/"+str(imgid[i])+".jpg")
             demo = Image.open(img_path+"/"+str(imgid[i])+".jpg").convert('RGB')
             self.demo_img = transform(demo)
             image1= Image.open(img_path+"/"+str(imgid[i])+".jpg")
             r, j= (image1.size)

             r, j = r / 64, j / 64
             b0 = ((self.bbox_num_img_cat[i][0]) / j)
             b1 = ((self.bbox_num_img_cat[i][1]) / r)
             b2 = ((self.bbox_num_img_cat[i][2]) / j)
             b3 = ((self.bbox_num_img_cat[i][3]) / r)
             self.bbox_num_img_cat[i] = [b0, b1, b2, b3]
             self.img_dict[imgid[i]]=self.demo_img,self.class_list_index,self.bbox_num_img_cat[i]#used cats_ann because there were duplicate id using cat_ids

         self.class_list_index += 1
         #print("the final", self.img_dict)

         image = cv2.imread(img_path+"/"+str(cats_ann_id[i])+".jpg")
        #plt.imshow(image)
        #plt.show()
        #print(b0,b1,b2,b3)
        image=cv2.rectangle(image, (int(b0), int(b1)), (int(b2 + b0), int(b3+b1)),(255,255,255) ,3)  # add rectangle to image
        #plt.axis('off')
        #plt.imshow(image)
          #cv2.imwrite('exp.jpg', image)
            #cv2.imshow("image",image)
        self.im = list(self.img_dict.values())


    def __getitem__(self,index):
        #dataset = datasets.ImageFolder(self.images, transform=self.transformations)
        return self.im[index][0] ,self.im[index][1],self.im[index][2]
    def __len__(self):
        self.len = len(self.img_dict)
        #self.dog_length = len(self.dog_images)
        return self.len
transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataserver_val=coco_downloader(args.root_path,args.class_list,args.images_per_class,args.coco_json_path_val,"Val",transform)
#total=len(dataserver_train)
val_dataloader = torch.utils.data.DataLoader(dataserver_val,batch_size=10, shuffle=False, num_workers=0)
class model(nn.Module):

    class SkipBlock(nn.Module):
                def __init__(self, in_ch, out_ch,middle, downsample=False, skip_connections=True):
                    super(model.SkipBlock, self).__init__()
                    self.downsample = downsample
                    self.skip_connections = skip_connections
                    self.convo1 = nn.Conv2d(in_ch, middle, 3, stride=1, padding=1)
                    self.convo2 = nn.Conv2d(middle, out_ch, 5, stride=1, padding=2)
                    self.convo3 = nn.Conv2d(out_ch, out_ch, 7, stride=1, padding=3)
                    self.convo4 = nn.Conv2d(out_ch, in_ch, 3, stride=1, padding=1)
                    norm_layer1 = nn.BatchNorm2d
                    norm_layer2 = nn.BatchNorm2d
                    self.bn1 = norm_layer1(32)
                    self.bn2 = norm_layer2(64)
                    self.bn3 = norm_layer2(3)
                    self.pool = nn.MaxPool2d(2, 2)
                    if downsample:
                        self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

                def forward(self, x):
                    identity = x
                    out = self.convo1(x)
                    out = torch.nn.functional.relu(out)
                    out = self.bn1(out)

                    out = self.convo2(out)

                    out = torch.nn.functional.relu(out)
                    out = self.bn2(out)

                    out = self.convo3(out)

                    out = torch.nn.functional.relu(out)
                    out = self.bn2(out)
                    out = self.convo4(out)
                    out = torch.nn.functional.relu(out)
                    out = self.bn3(out)

                    if self.skip_connections:
                            out += identity

                    return out

    class BB_model(nn.Module):

               def __init__(self):
                    super(model.BB_model, self).__init__()

                    #for classififcation
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fc1 = nn.Linear(3 * 16 * 16, 1000)  ## (C)
                    self.fc2 = nn.Linear(1000, 10)

                    self.skip64_arr = nn.ModuleList()

                    self.depth = 8
                    for i in range(self.depth):
                        self.skip64_arr.append(model.SkipBlock(3, 64,32, skip_connections=True))

                    ##  for regression
                    self.conv = nn.Conv2d(3, 64, 3, padding=1)
                    self.conv2 = nn.Conv2d(64,3, 3, padding=1)
                    self.conv_seqn = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    )
                    self.fc_seqn = nn.Sequential(
                        nn.Linear(16384, 1024),
                        nn.ReLU(inplace=True),
                        nn.Linear(1024, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 4)
                    )

                    resnet=models.resnet34(pretrained=True)
                    layers = list(resnet.children())[:8]
                    self.features1 = nn.Sequential(*layers[:6])
                    self.features2 = nn.Sequential(*layers[6:])
                    self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
                    self.bb = nn.Sequential(nn.BatchNorm1d(12288), nn.Linear(12288, 4096))

                    self.fc_seqn = nn.Sequential(
                       nn.Linear(4096, 1024),
                       nn.ReLU(inplace=True),
                       nn.Linear(1024, 512),
                       nn.ReLU(inplace=True),
                       nn.Linear(512, 4)
               )

               def forward(self, x):

                    ## The labeling section:

                    for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                         x = skip64(x)

                    ## The labeling section:
                    x=self.conv(x)

                    x= self.conv_seqn(x)
                    print(x.size)
                    x = self.conv2(x)

                    for i, skip64 in enumerate(self.skip64_arr[:self.depth // 4]):
                        x = skip64(x)

                    x = self.conv(x)
                    x = self.conv_seqn(x)
                    x = self.conv2(x)
                    for i, skip64 in enumerate(self.skip64_arr[:self.depth // 4]):
                        x = skip64(x)

                    x = self.conv(x)
                    x = self.conv_seqn(x)
                    x = self.conv2(x)

                    ## The labeling section:
                    x1 = self.pool(F.relu(self.conv(x)))
                    x1 = self.conv_seqn(x1)
                    x1 = self.conv2(x1)

                    x1 = self.pool(x1)


                    x1 = x1.view(-1, 3* 16 * 16)  ## (E)

                    x1 = F.relu(self.fc1(x1))
                    x1 = self.fc2(x1)

                    ## The Bounding Box regression:
                    for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                         x = skip64(x)

                    for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                         x = skip64(x)
                    for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                         x = skip64(x)

                    for i, skip64 in enumerate(self.skip64_arr[:self.depth // 4]):
                        x = skip64(x)

                    x2 = self.features1(x)
                    x2= self.features2(x2)
                    x2 = F.relu(x2)
                    x2 = nn.AdaptiveAvgPool2d((1, 1))(x2)
                    x2 = x.view(x2.shape[0], -1)
                    x2=self.bb(x2)
                    x2 = self.fc_seqn(x2)
                    return x1,x2

def train_epocs(model, train_dl, epoch=10, C=1000):
    dtype = torch.float64
    loss1 = []
    loss2 = []
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
    # dtype = torch.double64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    model = model.to(device)
    for i in range(epoch):
        model.train()
        running_loss_labeling = 0.0
        running_loss_regression = 0.0

        for i, data in enumerate(train_dl):
            x, y_class, y_bb = data
            x = x.to(device)

            y_class = (y_class).to(device)
            # y_bb=torch.FloatTensor(y_bb)
            y_bb = y_bb.float().to(device)
            out_class, out_bb = model(x)
            print(out_bb, y_bb)
            out_bb = (out_bb).to(device)
            optimizer.zero_grad()
            '''
            loss_class = F.cross_entropy(out_class, y_class)
            loss_class.backward()
            loss_bb = nn.MSELoss()((out_bb), (y_bb))
            print(loss_bb)
            loss_bb.backward()
            optimizer.step()

            running_loss_labeling += loss_class.item()
            running_loss_regression += loss_bb.item()
            if (i + 1) % 10 == 0:
                avg_loss_labeling = running_loss_labeling/10
                avg_loss_regression = running_loss_regression /10
                print("\n[epoch:%d, iteration:%5d]  loss_class: %.3f  loss_bb: %.3f  " % (epochs + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                running_loss_labeling = 0.0
                running_loss_regression = 0.0
            #val_loss, val_acc = val_metrics(model, valid_dl, C)
            '''
            loss_labeling = F.cross_entropy(out_class, y_class)
            loss_labeling.backward(retain_graph=True)
            loss_regression = nn.MSELoss()((out_bb), (y_bb))
            loss_regression.backward()
            optimizer.step()
            running_loss_labeling += loss_labeling.item()
            running_loss_regression += loss_regression.item()
            if i % 5 == 4:
                avg_loss_labeling = running_loss_labeling / float(5)
                avg_loss_regression = running_loss_regression / float(5)
                print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (
                    epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))

                running_loss_labeling = 0.0
                running_loss_regression = 0.0
                loss1.append(avg_loss_regression)
                loss2.append(avg_loss_labeling)

    torch.save(model.state_dict(), os.path.join(args.root_path + "val.pth"))
    plt.figure()
    plt.plot(loss1, color='blue', label="validation regression")
    plt.show()
    plt.plot(loss2, color='orange', label="validation classfication")
    plt.show()

BB_model = model.BB_model()

BB_model.load_state_dict(torch.load(os.path.join(args.root_path,"trainon.pth")))
train_epocs(BB_model, val_dataloader, epoch=20, C=args.images_per_class)

