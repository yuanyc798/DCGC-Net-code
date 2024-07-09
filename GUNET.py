# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:yyc

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1,dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=2,dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),)
        self.sent=senet(out_channels)
    def forward(self,x):
        x=self.double_conv(x)
        return x#self.sent(x)
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseConv,self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,kernel_size=3,padding=1,groups=in_channels),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),
            )
        #self.swish=Swish()
    def forward(self,x):
        x=self.d_conv(x)
        return x
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsample,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),#feature map 减半
            DoubleConv(in_channels,out_channels),
            Squdiat(out_channels,out_channels)
            ,)
    def forward(self, x):
        return self.maxpool_conv(x)
        
class downsamplS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsamplS,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),#feature map 减半
            DoubleConv(in_channels,out_channels)
            #Squdiat(in_channels,out_channels)
            ,)
    def forward(self, x):
        return self.maxpool_conv(x)
class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()
        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()
        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t 
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None] # B x C x 1 x 1
        z_hat = self.bn(z)
        g = self.activation(z_hat)
        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)
        # B x C x 1 x 1
        g = self._style_integration(t)
        return x * g                
class senet(nn.Module):
    def __init__(self, in_channels):
        super(senet,self).__init__()
        
        #self.globalpool= F.adaptive_avg_pool2d(xx,(1,1))
        self.line1=torch.nn.Linear(in_channels,int(in_channels//16))
        self.relu=nn.ReLU(inplace=True)
        self.line2=torch.nn.Linear(int(in_channels//16),in_channels)
        self.sigmoid=nn.Sigmoid()
        #self.reshape=torch.reshape()
        
    def forward(self, x):
        #print(x.shape)
        glb=F.adaptive_avg_pool2d(x,(1,1))

        glb=torch.squeeze(glb)
        #print(glb.shape)
        line1=self.line1(glb)
        relu=self.relu(line1)
        exc=self.line2(relu)
        #print(exc.shape)
        sigmoid=self.sigmoid(exc)
        exc=sigmoid.unsqueeze(-1)
        exc=exc.unsqueeze(-1)
        #print(exc.shape)
        out=torch.mul(x,exc)
        return out
        
class ATT(nn.Module):
    def __init__(self, in_channels):
        super(ATT,self).__init__()
        self.xcon2= nn.Conv2d(in_channels,1,kernel_size=1,padding=0)
        self.xcon3 = nn.Conv2d(1,1,kernel_size=3,padding=1)
        self.xcon5 = nn.Conv2d(1,1,kernel_size=5,padding=2)
        self.xcon7 = nn.Conv2d(1,1,kernel_size=7,padding=3)
        self.sigmoid=nn.Sigmoid()
        self.senet=senet(in_channels)
    def forward(self, x):
        sent=self.senet(x)
        x1=self.xcon2(x)
        #print('x1',x1.shape)
        x3=self.xcon3(x1)
        #print('x3',x3.shape)
        x5=self.xcon5(x1)
        #print('x5',x5.shape)
        x7=self.xcon7(x1)
        #print('x7',x7.shape)
        add1=torch.add(x3,x5)
        add2=torch.add(add1,x7)
        softm=self.sigmoid(add2)
        out1=torch.mul(softm,x)
        out=torch.mul(out1,sent)
        return out
class Squdiat(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Squdiat,self).__init__()
        self.xcon1=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,dilation=1),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.xcon2=nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=2,dilation=2),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.xcon3=nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=3,dilation=3),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.xcon5=nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=5,dilation=5),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))

        self.sigmoid=nn.Sigmoid()
        self.senet=senet(in_channels)
        self.spation=Spation(out_channels)
        self.xcon= nn.Conv2d(out_channels*4,out_channels,kernel_size=1,padding=0)
        self.srm=SRMLayer(out_channels*4) 
    def forward(self, x):
        #sent=self.senet(x)
        x1=self.xcon1(x)
        x2=self.xcon2(x1)
        #xb=x1+x2
        x3=self.xcon3(x2)
        #xb=xb+x3
        x5=self.xcon5(x3)
        #xb=xb+x5   
        xc=torch.cat([x1,x2,x3,x5],dim=1)
        #xc=self.srm(xc)
        xx=self.xcon(xc)     

        return xx

class RAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RAB,self).__init__()
        self.orgin=nn.Conv2d(in_channels, out_channels,kernel_size=1,padding=0,dilation=1)
        self.cconv =nn.Sequential(
            nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1,dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=3,dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=4,dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),)
        self.att=ATT(out_channels)
    def forward(self, x):
        #print('in',x.shape)
        
        cc=self.orgin(x)
        #print('cc',cc.shape)
        ee=self.cconv(cc)
        #print('ee',ee.shape)
        at=self.att(ee)
        #print('at',at.shape)
        add=torch.add(cc,at)
        return at

class Spation(nn.Module):
    def __init__(self, in_channels):
        super(Spation,self).__init__()
        self.xcon2= nn.Conv2d(in_channels,1,kernel_size=1,padding=0)
        self.xcon3 = nn.Conv2d(2,1,kernel_size=3,padding=1)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        maxx,e=torch.max(x,dim=1)
        #print(maxx.shape)
        meann=torch.mean(x,dim=1)
        #print(meann.shape)
        maxx=maxx.unsqueeze(1)
        meann=meann.unsqueeze(1)
        x1=self.xcon2(x)
        #print(x1.shape)
        #xc= torch.cat([maxx,meann,x1],dim=1)
        xc= torch.cat([maxx,meann],dim=1)
        xx=self.xcon3(xc)
        
        softm=self.sigmoid(xx)
        out=torch.mul(x,softm)
        return x+out



class upsample(nn.Module):
    def __init__(self, in_channels,out_channels,bilinear = True):
        super(upsample,self).__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels//2, out_channels//2,kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channels,out_channels)
        self.RAB=RAB(in_channels,out_channels)
        self.spation=Spation(2*out_channels)
        self.squt=Squdiat(in_channels,out_channels)
        self.convs= nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.depconv =DepthwiseConv(out_channels,out_channels)
    def forward(self,x1,x2):
        #x1 =self.convs(x1)
        x1 = self.upsample(x1)

        x = torch.cat([x2,x1],dim=1)
        #x=self.spation(x)
        x=self.conv(x)
        return x#self.squt(x)#self.spation(x)

class output_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(output_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.softmx=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.conv(x)
        return self.sigmoid(x)#self.softmx(x)

class GUNET(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear = True):
        super(GUNET,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        nmlist=[32,48,64,96,128,256]#[32,64,128,256,512,512]#
        self.init = DoubleConv(n_channels,nmlist[0])
        self.downsample1 = downsamplS(nmlist[0],nmlist[1])
        self.downsample2 = downsamplS(nmlist[1],nmlist[2])
        self.downsample3 = downsamplS(nmlist[2],nmlist[3])
        self.downsample4 = downsamplS(nmlist[3],nmlist[4])
        self.downsample5 = downsamplS(nmlist[4],nmlist[5])
        
        self.upsample0 = upsample(nmlist[5]+nmlist[4],nmlist[4],bilinear)#+nmlist[4]
        self.upsample1 = upsample(nmlist[4]+nmlist[3],nmlist[3],bilinear)#+nmlist[3]
        self.upsample2 = upsample(nmlist[3]+nmlist[2],nmlist[2],bilinear)#+nmlist[2]
        self.upsample3 = upsample(nmlist[2]+nmlist[1],nmlist[1],bilinear)#+nmlist[1]
        self.upsample4 = upsample(nmlist[1]+nmlist[0],nmlist[0],bilinear)#+nmlist[0]
        
        self.spation0=Spation(nmlist[4])
        self.spation1=Spation(nmlist[3])
        self.spation2=Spation(nmlist[2])
        self.spation3=Spation(nmlist[1])
        self.spation4=Spation(nmlist[0])
        
        self.squt=Squdiat(nmlist[5],nmlist[5])
        self.outconv = output_conv(nmlist[0],n_classes)
    def forward(self,x):
        x1 = self.init(x)
        #print(x1.shape)
        x2 = self.downsample1(x1)
        #print(x2.shape)
        x3 = self.downsample2(x2)
        #print(x3.shape)
        x4 = self.downsample3(x3)
        #print(x4.shape)
        x5 = self.downsample4(x4)
        #print(x5.shape)
        x6 = self.downsample5(x5)
        x6=self.squt(x6)
        
        #print(x6.shape,x5.shape)
        x = self.upsample0(x6,x5)
        #x=self.spation0(x)
        
        x = self.upsample1(x,x4)
        #x=self.spation1(x)
        
        x = self.upsample2(x,x3)
        #x=self.spation2(x)
        
        x = self.upsample3(x,x2)
        #x=self.spation3(x)
        
        x = self.upsample4(x,x1)
        #x=self.spation4(x)
        
        res = self.outconv(x)
        return res
#GUNET(3,2)(torch.randn(4,3, 320, 320))


