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
from q_pytorch_mri_tau_amyloid import AttentionModel

class MVBlock(nn.Module):
    def __init__(self):
        super(MVBlock, self).__init__()
        self.conv2d_1x = depthwise_separable_conv(64, 128)
        self.conv2d_2x = depthwise_separable_conv(128, 256)
        self.conv2dx = nn.Sequential(self.conv2d_1x, nn.InstanceNorm2d(128), nn.LeakyReLU(0.2) , \
                                    self.conv2d_2x, nn.InstanceNorm2d(256), nn.LeakyReLU(0.2))
        
        self.resblock_1x, self.resblock_2x = get_res_extend_block()
        self.resblock_x = nn.Sequential(self.resblock_1x, self.resblock_2x)

    def forward(self, x):
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

        res_featx = self.resblock_x(x)
        featx = torch.cat((cor_x, sag_x, axl_x, res_featx), dim=1)
        
        return featx


class ExtractNet(nn.Module):
    def __init__(self, teacher_path):
        super(ExtractNet, self).__init__()
        self.res_base = generate_model(10)
                
        #self.cbam = CBAM_Block(256*3+256)
        self.attx = NONLocalBlock3D(in_channels=256*3+256, bn_layer=True)
        self.atty = NONLocalBlock3D(in_channels=256*3+256, bn_layer=True)
        self.attz = NONLocalBlock3D(in_channels=256*3+256, bn_layer=True)

        self.mvx = MVBlock()
        self.mvy = MVBlock()
        self.mvz = MVBlock()

        #downsample = self._downsample_basic_block()
        self.bottle_neckx = DownScale(256*3+256+64, planes=8, stride=2)
        self.bottle_necky = DownScale(256*3+256+64, planes=8, stride=2)
        self.bottle_neckz = DownScale(256*3+256+64, planes=8, stride=2)

        self.pool = nn.AdaptiveMaxPool3d(8)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.surf_teacher = AttentionModel(dims=192,
                               depth=[3,1],
                               heads=3,
                               num_patches=640,
                               num_classes=1,
                               num_channels=4,
                               num_vertices=153,
                               dropout=0.1,
                               branches=[slice(0, 3), slice(3, 5)],
                               activation='sigmoid').to(device)
        
        
        self.surf_teacher.load_state_dict(torch.load(teacher_path, map_location=device), strict=False)
        
        '''freeze teacher's weights'''
        for name, param in self.surf_teacher.named_parameters():
            param.requires_grad = False

        self.fc_fuse_att_surfx = nn.Linear(640*640, 128)
        self.fc_fuse_v_surfx = nn.Linear(640*64, 128)
        self.fc_att_weight_imgx = nn.Linear(512*128, 128)
        self.fc_att_value_imgx = nn.Linear(128*512, 128)

        self.fc_fuse_att_surfy = nn.Linear(640*640, 128)
        self.fc_fuse_v_surfy = nn.Linear(640*64, 128)
        self.fc_att_weight_imgy = nn.Linear(512*128, 128)
        self.fc_att_value_imgy = nn.Linear(128*512, 128)

        self.fc_fuse_att_surfz = nn.Linear(640*640, 128)
        self.fc_fuse_v_surfz = nn.Linear(640*64, 128)
        self.fc_att_weight_imgz = nn.Linear(512*128, 128)
        self.fc_att_value_imgz = nn.Linear(128*512, 128)
    

    def forward(self, x, y, z, surfs):
        
        bs = x.shape[0]

        x = self.res_base(x) #shape: (bs,  64, 45, 45, 45)
        y = self.res_base(y)
        z = self.res_base(z)

        featx = self.mvx(x)
        featy = self.mvy(y)
        featz = self.mvz(z)
               

        featx = self.pool(featx)
        featy = self.pool(featy)
        featz = self.pool(featz)


        featx, att_weight_imgx, att_value_imgx = self.attx(featx)
        featx = torch.cat([featx, self.pool(x)], dim=1)
        featx = self.bottle_neckx(featx)

        featy, att_weight_imgy, att_value_imgy = self.atty(featy)
        featy = torch.cat([featy, self.pool(y)], dim=1)
        featy = self.bottle_necky(featy)

        featz, att_weight_imgz, att_value_imgz = self.attz(featz)
        featz = torch.cat([featz, self.pool(z)], dim=1)
        featz = self.bottle_neckz(featz)


        #---------------------surf----------------------------------------------------
        out_surf, fuse_att_surfx, fuse_v_surfx, fuse_att_surfy, fuse_v_surfy, fuse_att_surfz, \
            fuse_v_surfz = self.surf_teacher(surfs)


        att_weight_imgx = att_weight_imgx.contiguous().view(bs, -1)#torch.mean(att_weight_img, dim=1)
        fuse_att_surfx = fuse_att_surfx.contiguous().view(bs, -1)#torch.mean(fuse_att_surf, dim = 1)
        att_value_imgx = att_value_imgx.contiguous().view(bs, -1)
        fuse_v_surfx = fuse_v_surfx.contiguous().view(bs, -1)


        att_weight_imgy = att_weight_imgy.contiguous().view(bs, -1)#torch.mean(att_weight_img, dim=1)
        fuse_att_surfy = fuse_att_surfy.contiguous().view(bs, -1)#torch.mean(fuse_att_surf, dim = 1)
        att_value_imgy = att_value_imgy.contiguous().view(bs, -1)
        fuse_v_surfy = fuse_v_surfy.contiguous().view(bs, -1)

        att_weight_imgz = att_weight_imgz.contiguous().view(bs, -1)#torch.mean(att_weight_img, dim=1)
        fuse_att_surfz = fuse_att_surfz.contiguous().view(bs, -1)#torch.mean(fuse_att_surf, dim = 1)
        att_value_imgz = att_value_imgz.contiguous().view(bs, -1)
        fuse_v_surfz = fuse_v_surfz.contiguous().view(bs, -1)


        return featx, featy, featz, self.fc_fuse_att_surfx(fuse_att_surfx), self.fc_fuse_v_surfx(fuse_v_surfx), \
            self.fc_fuse_att_surfy(fuse_att_surfy), self.fc_fuse_v_surfy(fuse_v_surfy), \
            self.fc_fuse_att_surfz(fuse_att_surfz), self.fc_fuse_v_surfz(fuse_v_surfz), \
            self.fc_att_weight_imgx(att_weight_imgx), self.fc_att_value_imgx(att_value_imgx), \
            self.fc_att_weight_imgy(att_weight_imgy), self.fc_att_value_imgy(att_value_imgy), \
            self.fc_att_weight_imgz(att_weight_imgz), self.fc_att_value_imgz(att_value_imgz)


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
        self.fc1 = nn.Linear(8*(4**3)*3, 32)
        self.lkrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(0.4)
        self.drop1 = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(8*(4**3)*3 ,eps=1e-6)
        self.sig1 = nn.Sigmoid()
        
        self.clf1 = SubClf()
        self.clf2 = SubClf()
        self.clf3 = SubClf()

    def forward(self, x, y, z, surfs):
        featx, featy, featz, fuse_att_surfx, fuse_v_surfx, fuse_att_surfy, fuse_v_surfy, \
            fuse_att_surfz, fuse_v_surfz, \
            att_weight_imgx, att_value_imgx, att_weight_imgy, att_value_imgy, \
                att_weight_imgz, att_value_imgz = self.extract_net(x, y, z, surfs)

        x = self.drop(featx)
        y = self.drop(featy)
        z = self.drop(featz)

        latentx = self.avg(x)
        x_fl = latentx.view(latentx.size(0), -1)

        latenty = self.avg(y)
        y_fl = latenty.view(latenty.size(0), -1)

        latentz = self.avg(z)
        z_fl = latentz.view(latentz.size(0), -1)

        feat = torch.cat([x_fl, y_fl, z_fl], dim=1)
        feat = self.norm(feat)

        x = self.fc1(feat)
        x = x*self.sig1(x)
        x = self.lkrelu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.sig(x)

        x_fl = self.clf1(x_fl)
        y_fl = self.clf2(y_fl)
        z_fl = self.clf3(z_fl)

        
        return x, x_fl, y_fl, z_fl, fuse_att_surfx, fuse_v_surfx, fuse_att_surfy, \
            fuse_v_surfy, fuse_att_surfz, fuse_v_surfz, att_weight_imgx, att_value_imgx, \
                att_weight_imgy, att_value_imgy, att_weight_imgz, att_value_imgz