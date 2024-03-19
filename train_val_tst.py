import os
import sys
from tqdm import tqdm
#from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#from torchvision.utils import make_grid
from torch.nn.modules.loss import CrossEntropyLoss,BCEWithLogitsLoss,BCELoss

from dataset import PLQ_Dataset
from discriminator import FCDiscriminator
import ramps, losses
from sem_process import  RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import math
import random
from GUNET import *
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=100.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path

from glob import glob
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 1


def dice_m(score, target, smooth=1e-10):
    #score =torch.argmax(score,dim=1).float()
    score=score[:,0,:,:]    
    target=target[:,0,:,:]#target =torch.argmax(target,dim=1)

    intersect = torch.sum(score * target,dim=(1,2))
    y_sum = torch.sum(target * target,dim=(1,2))
    #print(torch.sum(y_sum).item())
    z_sum = torch.sum(score * score,dim=(1,2))
    dc= (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return torch.mean(dc.float())
def preprocess_input(x):#BGR
    #x = skimage.color.rgb2gray(x) 
    x = (x - np.mean(x)) / np.std(x)
    return x
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
def validate(net, valid_loader,batch_size):
        # validation steps
        with torch.no_grad():
            net.eval()
            valid_dc = 0
            loss = 0
            for images, labels in valid_loader:
                images = images.cuda().float()
                labels = labels.cuda().float()
                pred = net(images)
                #pred=F.softmax(pred,dim=1)
                dc = dice_m(pred,labels)
                valid_dc+=dc
                #print(dc)
                loss += dice_loss(pred,labels)
        return valid_dc/len(valid_loader),loss/len(valid_loader)
        
def test_net(net,traintime,tst_path):
        state_dict = torch.load('./save/best_model.pth')
        net.load_state_dict(state_dict)
        #tst_path='/storage/yyc/data/tst/'
        listname=glob(tst_path+'*.jpg')
        path_save=r'./save/Gunt/'
        isExists=os.path.exists(path_save)
        if not isExists:
           os.makedirs(path_save)
        
        with torch.no_grad():
            net.eval()
            tc1=time.time()
            for image_path in listname:
                image = cv2.imread(image_path)
                #X =preprocess_input(image)
                image = np.array(image)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                image=preprocess_input(image)
                image=image.reshape(1,image.shape[0],image.shape[1])
                image=np.expand_dims(image, axis=0)
                image=torch.from_numpy(image)
                mg=image.cuda().float()
                 
                y1=net(mg)
                mgn=y1.cpu()
                mgn=torch.squeeze(mgn)
                #print(mgn.shape) 
                mm=mgn.numpy()
                mm[mm>=0.5]=1
                mm[mm<0.5]=0
                cv2.imwrite(path_save+image_path.split('/')[-1].split('.')[0]+'.png',mm*255)
            tc2=time.time()  
        tsttime=(tc2-tc1)/400      
        f = open(path_save+"time.txt",'w')
        f.write('train time:')
        f.write(str(traintime))
        f.write('\n')
        f.write('tst time:')
        f.write(str(tsttime))
        f.write('\n')         
        f.close() 
                
def rotate2(mg,mode):
    ang=[-90,90,30,-30]
    img=mg.cpu().float()
    batch_size,out_h,out_w=img.shape[0],img.shape[2],img.shape[3]
    angle =ang[mode]* math.pi / 180  # 
    A = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    tx,ty = 0,0
    theta = np.array([[A[0, 0], A[0, 1], tx], [A[1, 0], A[1, 1], ty]])
    theta = torch.from_numpy(theta).float().unsqueeze(0).repeat(batch_size,1,1)
    out_size = torch.Size((batch_size,1, out_h, out_w))
    
    grid = F.affine_grid(theta, out_size)
    warped_image_batch = F.grid_sample(img, grid)
    return mg#warped_image_batch.cuda().float()
def rotate(mg,mode):
    mode=1
    return mg    
device = torch.device('cuda:0')
if __name__ == "__main__":

    def create_model(ema=False):
        net =GUNET(1,1)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model=create_model()
    ema_model= create_model(ema=True)

    D= FCDiscriminator(num_classes=1)
    D= D.cuda()

    db_train= PLQ_Dataset(train_data_path+'train/',True,True)                   
    db_val= PLQ_Dataset(train_data_path+'val/',True,False)
        
    labeled_idxs = list(range(400))
    unlabeled_idxs = list(range(400,800))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader=DataLoader(db_val,batch_size,shuffle = False)
    
    model.train()
    ema_model.train()
    #Dopt= optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.99))
    Dopt= optim.SGD(D.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0001)
    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer =optim.Adam(model.parameters(),lr=base_lr, betas=(0.9, 0.999), eps=1e-08)
    ce_loss =BCELoss()#F.cross_entropy#CrossEntropyLoss()#()#BCEWithLogitsLoss()#
    dice_loss =losses.dc_loss#losses.DiceLoss(2)#DiceLoss(2)
    
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)
    lr_ = base_lr
    model.train()
    val_dice=0
    val_ls=1
    epochs=max_epoch#200
    t1=time.time()
    for epoch_num in tqdm(range(epochs), ncols=100):
        time1 = time.time()
        lossm=0
        lossD=0
        for  i,(image,label) in enumerate(trainloader):
            time2 = time.time()
            mode = random.choice([0,1,2,3])
            
            volume_batch, label_batch = image,label#sampled_batch['image'], sampled_batch['label']

            Dtarget = torch.tensor([1,1,1,1,0,0,0,0]).cuda()
            #Dtarget = torch.tensor([1,1,0,0]).cuda()
            model.train()
            D.eval()
            volume_batch, label_batch =volume_batch.cuda().float(),label_batch.cuda().float()
            
            #print(volume_batch)
            label_batch=torch.squeeze(label_batch,1)
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            
            outputs = model(volume_batch)

            #outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                #ema_output_soft = torch.softmax(ema_output, dim=1)

            #print(outputs[:args.labeled_bs,0,:,:].shape,label_batch[:args.labeled_bs].shape)
            loss_ce = ce_loss(outputs[:args.labeled_bs,0,:,:],label_batch[:args.labeled_bs])
            #print(outputs_soft.shape,label_batch.shape)            
            loss_dice = dice_loss(outputs[:args.labeled_bs,0,:,:], label_batch[:args.labeled_bs])
                
            supervised_loss=1*loss_dice#+0.3*loss_ce
            consistency_weight = get_current_consistency_weight(iter_num//150)

            consistency_loss = torch.mean((outputs[args.labeled_bs:]-ema_output)**2)
            
            Doutputs = D(outputs[labeled_bs:], volume_batch[labeled_bs:])
            #Dta= torch.tensor([1,1,1,1,1,1]).cuda()
            loss_adv = F.cross_entropy(Doutputs,Dtarget[:labeled_bs].long())#Dta.long()
            conEM_loss = losses.entropy_loss(outputs, C=2)

            weight=0.1                          
            loss = supervised_loss+consistency_weight * (consistency_loss)+weight*loss_adv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossm+=loss.item()
            
            
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            #for param_group in optimizer.param_groups:
                #param_group['lr'] = lr_

            model.eval()#self.train(False) not training
            D.train()
            with torch.no_grad():
                outputs = model(volume_batch)

            #label_batch=torch.unsqueeze(label_batch,1)
            #outputs=torch.cat([label_batch[0:3],outputs],dim=0)
            #volume_b=torch.cat([volume_batch[0:3],volume_batch],dim=0)
              
            Doutputs = D(outputs, volume_batch)
            # D want to classify unlabel data and label data rightly.
            #print(Doutputs.shape,Dtarget.shape)
            
            D_loss = F.cross_entropy(Doutputs, Dtarget.long())
            Dopt.zero_grad()
            D_loss.backward()
            Dopt.step()
            lossD+=D_loss.item()
            
            iter_num = iter_num + 1
        v_dc,val_loss=validate(model,val_loader,batch_size)
        print('epoch:%d  train loss:%f' % (epochs+1,lossm/len(trainloader)),'  train Dloss:',lossD/len(trainloader))
        if v_dc >val_dice:
                print('val_dice changed:',v_dc.item(),'model saved, val_loss:',val_loss.item())
                #val_dice=v_dc
                val_dice=v_dc
                torch.save(model.state_dict(), './save/best_model.pth')
    t2=time.time() 
    t=t2-t1
    tst_path=r'./data/tst/'
    test_net(model,t/3600,tst_path)
    

