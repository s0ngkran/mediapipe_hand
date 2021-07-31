# using pre-trained VGG
from torchvision import transforms
import torch
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

def make_vgg(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                     kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))

class HandModel(torch.nn.Module):
    def __init__(self, input_channel):
        super(HandModel, self).__init__()
        assert input_channel in [1,3]
        no_relu_layers = []
        vgg = OrderedDict([
                      ('conv1_1', [input_channel, 64, 3, 1, 1]), 
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ])
        self.vgg = make_vgg(vgg, no_relu_layers)
        '''
        transfer weight

        vgg16 features
        0 torch.Size([64, 3, 3, 3])
        1 ReLU
        2 torch.Size([64, 64, 3, 3])
        3 ReLU
        4 MaxPool2d
        5 torch.Size([128, 64, 3, 3])
        6 ReLU
        7 torch.Size([128, 128, 3, 3])
        8 ReLU
        9 MaxPool2d
        10 torch.Size([256, 128, 3, 3])
        11 ReLU
        12 torch.Size([256, 256, 3, 3])
        13 ReLU
        14 torch.Size([256, 256, 3, 3])
        15 ReLU
        16 MaxPool2d
        17 torch.Size([512, 256, 3, 3])
        18 ReLU
        19 torch.Size([512, 512, 3, 3])
        20 ReLU

        '''
        assert input_channel == 3
        trained_vgg = models.vgg16(pretrained=True)
        for i in range(0, 21):
            trained_feature = trained_vgg.features[i]
            self_vgg_feature = self.vgg[i]
            layer_name_trained = type(trained_feature).__name__
            if layer_name_trained.startswith('Conv2d'):
                print('trained transfer...', trained_feature.weight.shape, '->', self_vgg_feature.weight.shape)
                self_vgg_feature.weight = trained_feature.weight
                self_vgg_feature.bias = trained_feature.bias
                for param in self_vgg_feature.parameters():
                    param.requires_grad = False

        no_relu_layers = []
        _ = OrderedDict([
            ('conv4_3_CPM', [512, 256, 3, 1, 1]),
            ('conv4_4_CPM', [256, 128, 3, 1, 1])
        ])
        self.vgg_fine_tuning = make_vgg(_, no_relu_layers)
            
        self.drop = nn.Dropout(p=0.2)
        
        out_channel_from_vgg = 128
        gtl_size = 22
        gts_size = 12
        self.L1 = Stage(out_channel_from_vgg, gtl_size)
        self.L2 = Stage(out_channel_from_vgg + gtl_size, gtl_size)
        self.L3 = Stage(out_channel_from_vgg + gtl_size, gtl_size)
        # self.L3 = Stage(130, 2, no_relu=True)
        # self.L4 = Stage(148, 20)
        # self.L5 = Stage(148, 20)
        # self.L6 = Stage(148, 20)

        self.S1 = Stage(out_channel_from_vgg + gtl_size, gts_size) 
        self.S2 = Stage(out_channel_from_vgg + gtl_size + gts_size, gts_size) # 159 = 128+20+11
        self.S3 = Stage(out_channel_from_vgg + gtl_size + gts_size, gts_size) # 159 = 128+20+11
        # self.S3 = Stage(159, 11)
        # self.S4 = Stage(159, 11)
        # self.S5 = Stage(159, 11)
        # self.S6 = Stage(159, 11)
        
    def forward(self, x):
        # fix weight
        pre_trained_vgg = self.vgg(x)

        # running weight
        features = self.vgg_fine_tuning(pre_trained_vgg)
        features = self.drop(features)

        L1 = self.L1(features) # 256
        L1 = torch.tanh(L1)
        S1 = self.S1(torch.cat([features, L1], dim=1)) # 256 + 46
        S1 = torch.sigmoid(S1) 
        
        # L2 = self.L2(torch.cat([features, L1], dim=1)) # 256 + 46
        # L2 = torch.tanh(L2)
        # S2 = self.S2(torch.cat([features, L2, S1], dim=1)) # 256 + 46 + 25
        # S2 = torch.sigmoid(S2)

        # L3 = self.L3(torch.cat([features, L2], dim=1)) # 256 + 46
        # L3 = torch.tanh(L3)
        # S3 = self.S3(torch.cat([features, L3, S2], dim=1)) # 256 + 46 + 25
        # S3 = torch.sigmoid(S3)

        # if torch.isnan(S2):
        #     with open('check_nan.txt', 'w') as f:
        #         f.write('L1=')
        #         f.write(str(L1))
        #         f.write('L2=')
        #         f.write(str(L2))
        #         f.write('S1=')
        #         f.write(str(S1))
        #         f.write('S2=')
        #         f.write(str(S2))
        
        # L2 = self.L2(torch.cat([Fea, L1], dim=1))
        # L3 = self.L3(torch.cat([Fea, L2], dim=1))
        # S3 = self.S3(torch.cat([Fea, L3, S2], dim=1))
        # L4 = self.L4(torch.cat([Fea, L3], dim=1))
        # S4 = self.S4(torch.cat([Fea, L4, S3], dim=1))
        # L5 = self.L5(torch.cat([Fea, L4], dim=1))
        # S5 = self.S5(torch.cat([Fea, L5, S4], dim=1))
        # L6 = self.L6(torch.cat([Fea, L5], dim=1))
        # S6 = self.S6(torch.cat([Fea, L6, S5], dim=1))    
        # return L1, L2, L3, L4, L5, L6, S1, S2, S3, S4, S5, S6
      
        # return (S1, S2, S3), (L1, L2, L3)
        return S1, L1

class Stage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stage, self).__init__()
        self.module1 = Inception(in_channels)
        self.module2 = Inception(384) # 128*3 = 384
        self.module3 = Inception(384)
        self.module4 = Inception(384)
        self.module5 = Inception(384)
    
        self.c1 = Conv( 384, 128, 1, 1, 0)
        self.c2 = ConvWithBN( 128, out_channels, 1, 1, 0) 

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.module5(x)
        x = self.c1(x)
        x = self.c2(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.c1 = Conv(in_channels, 128, 3, 1, 1)
        self.c2 = Conv(128, 128, 3, 1, 1)
        self.c3 = Conv(128, 128, 3, 1, 1)
    def forward(self, x):
        y1 = self.c1(x)
        y2 = self.c2(y1)
        y3 = self.c3(y2)
        return torch.cat([y1, y2, y3], dim=1)
class Conv_leaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv_leaky, self).__init__()
        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.ReLU(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class ConvWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def test_forword_with_example_image():
    import urllib
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    from PIL import Image
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    print('image shape', input_batch.shape)
    model = HandModel(3)
    outputs = model(input_batch)
    for output in outputs:
        for stage in output:
            print(stage[0].shape)

def test_forward():
    model = HandModel(3)
    img = torch.rand([12,3,480,480]).type(torch.float32)
    print('input=',img.shape)
    output = model(img)
    print(output.shape)
    # for mode in output:
    #     print(mode[0].shape)


def check_model_size():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('before')
    print('total_memory =', t)
    print('reserved =', r)
    print('allocated =', a)
    print('free =', f)
    
    channel = 1
    model = HandModel(channel).cuda()
    
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('after')
    print('total_memory =', t)
    print('reserved =', r)
    print('allocated =', a)
    print('free =', f)
    input('pause')
    
if __name__ =='__main__':
    # test_forward()
    test_forword_with_example_image()

