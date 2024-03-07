import torch
import torch.nn as nn
import torch.nn.functional as F


class FC3DDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(FC3DDiscriminator, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(
            num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(
            n_channel, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv3d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(
            ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(
            ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((6, 6, 6))  # (D/16, W/16, H/16)
        self.classifier = nn.Linear(ndf*8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()

    def forward(self, map, image):
        batch_size = map.shape[0]
        map_feature = self.conv0(map)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)

        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x
class NONLocal(nn.Module):
    """
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """
    def __init__(self,in_channels,inter_channels=None,dimension=2,sub_sample=True,
                 bn_layer=True):
        super(NONLocal, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,out_channels=self.inter_channels,
                         kernel_size=1,stride=1,padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,kernel_size=1,stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,out_channels=self.in_channels,
                             kernel_size=1,stride=1,padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,kernel_size=1,
                             stride=1,padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,kernel_size=1,stride=1,padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        #print('x',f.shape)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(FCDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)           
        self.conv1 = nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
        
        self.conv = nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
                    
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
            
        self.liner = nn.Linear(ndf*32,ndf*4)
        self.liner2 = nn.Linear(ndf*4,ndf)            
        self.classifier = nn.Linear(ndf, 2)
        self.avgpool = nn.AvgPool2d((10, 10))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()
        self.nonloc=NONLocal(ndf*8)
        
    def forward(self, map, feature):
        map_feature = self.conv0(map)        
        image_feature = self.conv1(feature)
        
        #x = torch.add(map,feature)
        #x= self.conv(x)
        x = torch.add(map_feature, image_feature)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        #x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        #x = self.dropout(x)

        x = self.conv4(x)
        #x=self.nonloc(x)
        
        x = self.leaky_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x =self.liner(x)
        x = self.leaky_relu(x)
        x =self.liner2(x) 
        x = self.leaky_relu(x)
                
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x
