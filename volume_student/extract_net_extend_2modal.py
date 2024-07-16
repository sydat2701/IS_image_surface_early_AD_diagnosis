import torch
import torch.nn as nn
from components.resbased_block import generate_model
from components.depthwise_sep_conv import depthwise_separable_conv
from components.cbam import CBAM_Block
from components.resbase_extend_block import get_res_extend_block
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
class DownScale(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=False):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes)
        self.bn3 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if self.downsample:
        #     residual = F.avg_pool3d(x, kernel_size=1, stride=self.stride)

        # out += residual
        out = self.relu(out)

        return out

from teacher.NL_block import NONLocalBlock3D
from q_pytorch_mri_fdg import AttentionModel


class ExtractNet(nn.Module):
    def __init__(self, teacher_path):
        super(ExtractNet, self).__init__()
        self.res_base = generate_model(10)
        
        self.conv2d_1x = depthwise_separable_conv(64, 128)
        self.conv2d_2x = depthwise_separable_conv(128, 256)

        self.conv2d_1y = depthwise_separable_conv(64, 128)
        self.conv2d_2y = depthwise_separable_conv(128, 256)

        self.conv2dx = nn.Sequential(self.conv2d_1x, nn.InstanceNorm2d(128), nn.LeakyReLU(0.2) , \
                                    self.conv2d_2x, nn.InstanceNorm2d(256), nn.LeakyReLU(0.2))
        
        self.conv2dy = nn.Sequential(self.conv2d_1y, nn.InstanceNorm2d(128), nn.LeakyReLU(0.2) , \
                                    self.conv2d_2y, nn.InstanceNorm2d(256), nn.LeakyReLU(0.2))
        
        #self.cbam = CBAM_Block(256*3+256)
        self.attx = NONLocalBlock3D(in_channels=256*3+256, bn_layer=True)
        self.atty = NONLocalBlock3D(in_channels=256*3+256, bn_layer=True)

        self.resblock_1x, self.resblock_2x = get_res_extend_block()
        self.resblock_1y, self.resblock_2y = get_res_extend_block() 

        self.resblock_x = nn.Sequential(self.resblock_1x, self.resblock_2x)
        self.resblock_y = nn.Sequential(self.resblock_1y, self.resblock_2y)

        #downsample = self._downsample_basic_block()
        self.bottle_neckx = DownScale(256*3+256+64, planes=8, stride=2)
        self.bottle_necky = DownScale(256*3+256+64, planes=8, stride=2)

        self.pool = nn.AdaptiveMaxPool3d(8)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.surt_teacher = AttentionModel(dims=192,
                               depth=[3,1],
                               heads=3,
                               num_patches=640,
                               num_classes=1,
                               num_channels=4,
                               num_vertices=153,
                               dropout=0.1,
                               branches=[slice(0, 3), slice(3, 4)],
                               activation='sigmoid').to(device)
        
        
        self.surt_teacher.load_state_dict(torch.load(teacher_path, map_location=device), strict=False)
        
        '''freeze teacher's weights'''
        for name, param in self.surt_teacher.named_parameters():
            param.requires_grad = False

        self.fc_fuse_att_surfx = nn.Linear(640*640, 128)
        self.fc_fuse_v_surfx = nn.Linear(640*64, 128)
        self.fc_att_weight_imgx = nn.Linear(512*128, 128)
        self.fc_att_value_imgx = nn.Linear(128*512, 128)

        self.fc_fuse_att_surfy = nn.Linear(640*640, 128)
        self.fc_fuse_v_surfy = nn.Linear(640*64, 128)
        self.fc_att_weight_imgy = nn.Linear(512*128, 128)
        self.fc_att_value_imgy = nn.Linear(128*512, 128)
    

    def forward(self, x, y, surfs):
        

        x = self.res_base(x) #shape: (bs,  64, 45, 45, 45)
        y = self.res_base(y)
        
        bs, c, h, t, w = x.shape
        cor_x = x.permute(0, 3, 1, 2, 4).contiguous().view(bs*t, c, h, w) 
        sag_x = x.permute(0, 4, 1, 2, 3).contiguous().view(bs*w, c, h, t)
        axl_x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs*h, c, t, w)

        cor_x = self.conv2dx(cor_x)
        cor_x = cor_x.view(bs, cor_x.size(1), -1, cor_x.size(2), cor_x.size(3)) #(bs, c, t, h, w)
        sag_x = self.conv2dx(sag_x)
        sag_x = sag_x.view(bs, sag_x.size(1), -1, sag_x.size(2), sag_x.size(3))
        axl_x = self.conv2dx(axl_x)
        axl_x = axl_x.view(bs, axl_x.size(1), -1, axl_x.size(2), axl_x.size(3))


        bs, c, h, t, w = y.shape
        cor_y = y.permute(0, 3, 1, 2, 4).contiguous().view(bs*t, c, h, w) 
        sag_y = y.permute(0, 4, 1, 2, 3).contiguous().view(bs*w, c, h, t)
        axl_y = y.permute(0, 2, 1, 3, 4).contiguous().view(bs*h, c, t, w)

        cor_y = self.conv2dy(cor_y)
        cor_y = cor_y.view(bs, cor_y.size(1), -1, cor_y.size(2), cor_y.size(3)) #(bs, c, t, h, w)
        sag_y = self.conv2dy(sag_y)
        sag_y = sag_y.view(bs, sag_y.size(1), -1, sag_y.size(2), sag_y.size(3))
        axl_y = self.conv2dy(axl_y)
        axl_y = axl_y.view(bs, axl_y.size(1), -1, axl_y.size(2), axl_y.size(3))

        res_featx = self.resblock_x(x)
        res_featy = self.resblock_y(y)

        featx = torch.cat((cor_x, sag_x, axl_x, res_featx), dim=1)
        featy = torch.cat((cor_y, sag_y, axl_y, res_featy), dim=1)

        featx = self.pool(featx)
        featy = self.pool(featy)

        # featx = self.attx(featx)
        # featy = self.atty(featy)

        featx, att_weight_imgx, att_value_imgx = self.attx(featx)
        featx = torch.cat([featx, self.pool(x)], dim=1)
        featx = self.bottle_neckx(featx)

        featy, att_weight_imgy, att_value_imgy = self.atty(featy)
        featy = torch.cat([featy, self.pool(y)], dim=1)
        featy = self.bottle_necky(featy)

        #---------------------surf----------------------------------------------------
        out_surf, fuse_att_surfx, fuse_v_surfx, fuse_att_surfy, fuse_v_surfy = self.surt_teacher(surfs)

        
        #-----------------------------------------------------------------------------

        att_weight_imgx = att_weight_imgx.contiguous().view(bs, -1)#torch.mean(att_weight_img, dim=1)
        fuse_att_surfx = fuse_att_surfx.contiguous().view(bs, -1)#torch.mean(fuse_att_surf, dim = 1)
        att_value_imgx = att_value_imgx.contiguous().view(bs, -1)
        fuse_v_surfx = fuse_v_surfx.contiguous().view(bs, -1)


        att_weight_imgy = att_weight_imgy.contiguous().view(bs, -1)#torch.mean(att_weight_img, dim=1)
        fuse_att_surfy = fuse_att_surfy.contiguous().view(bs, -1)#torch.mean(fuse_att_surf, dim = 1)
        att_value_imgy = att_value_imgy.contiguous().view(bs, -1)
        fuse_v_surfy = fuse_v_surfy.contiguous().view(bs, -1)


        return featx, featy, self.fc_fuse_att_surfx(fuse_att_surfx), self.fc_fuse_v_surfx(fuse_v_surfx), \
            self.fc_fuse_att_surfy(fuse_att_surfy), self.fc_fuse_v_surfy(fuse_v_surfy), \
            self.fc_att_weight_imgx(att_weight_imgx), self.fc_att_value_imgx(att_value_imgx), \
            self.fc_att_weight_imgy(att_weight_imgy), self.fc_att_value_imgy(att_value_imgy)


class SubClf(nn.Module):
    def __init__(self):
        super(SubClf, self).__init__()
        self.fc1 = nn.Linear(8*(4**3), 32)
        self.fc2 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        self.lkrelu = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.lkrelu(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x

class MVNet(nn.Module):
    def __init__(self, teacher_path):
        super(MVNet, self).__init__()
        self.extract_net = ExtractNet(teacher_path)

        self.avg = nn.AdaptiveAvgPool3d(4)
        self.fc1 = nn.Linear(8*(4**3)*2, 32)
        self.lkrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(0.4)
        self.drop1 = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(8*(4**3)*2 ,eps=1e-6)
        self.sig1 = nn.Sigmoid()
        
        self.clf1 = SubClf()
        self.clf2 = SubClf()

    def forward(self, x, y, surfs):
        featx, featy, fuse_att_surfx, fuse_v_surfx, fuse_att_surfy, fuse_v_surfy, \
            att_weight_imgx, att_value_imgx, att_weight_imgy, att_value_imgy = self.extract_net(x, y, surfs)

        x = self.drop(featx)
        y = self.drop(featy)

        latentx = self.avg(x)
        x_fl = latentx.view(latentx.size(0), -1)

        latenty = self.avg(y)
        y_fl = latenty.view(latenty.size(0), -1)

        feat = torch.cat([x_fl, y_fl], dim=1)
        feat = self.norm(feat)

        x = self.fc1(feat)
        x = x*self.sig1(x)
        x = self.lkrelu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.sig(x)

        x_fl = self.clf1(x_fl)
        y_fl = self.clf2(y_fl)

        
        return x, x_fl, y_fl, fuse_att_surfx, fuse_v_surfx, fuse_att_surfy, \
            fuse_v_surfy, att_weight_imgx, att_value_imgx, att_weight_imgy, att_value_imgy