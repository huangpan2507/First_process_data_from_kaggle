# Kaggle-
这事关于对kaggle数据集的处理，使数据能够喂给网络。另外有关于数据处理的 https://pytorch.org/tutorials/beginner/data_loading_tutorial.html。

1. 训练集和测试集的数据读取
train=pd.read_csv(‘train.csv’,index_col=0)
test=pd.read_csv(‘test.csv’,index_col=0)
y_test=pd.read_csv(‘sample_submission.csv’,index_col=0)

2.数据合并

3.删除缺失样本数据

4.将数值型和字符型特征值分开


5.pd.get_dummies（数据集）函数 进行onehot哑编码       此处： # label_Onehot = pd.get_dummies(landmarks_frame['Type1'])  [0,0,0,0....1,0]只有一                  
                                                             个为1 

6.数据分割(之前把训练集和测试集合并在一起了，现在把他们分开）





import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import torch

import torch.nn as nn
import torchsnooper
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

from torch import optim
from PIL import Image
from torch.autograd import Variable
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


# Ingore warning
import warnings
warnings.filterwarnings('ignore')

plt.ion()        # 交互模式，遇到plt.show()，代码会继续执行，并不要等你关掉显示的图片，展示动态图或多个窗口

# 读取数据文件，image_name, return DataFrame or TextParser
# landmarks_frame = pd.read_csv('D:/Firefox_Download/pokemon_images/pokemon.csv')

# n = 243
# img_name = landmarks_frame.iloc[1:n, 0] + '.jpg'
# landmarks = landmarks_frame.iloc[1:n, 1:2]

# print('Image name: {}'.format(img_name))
# print('First 243 Lanmarks: {}'.format(landmarks[:]))

old_root_dir = '/home/huangpan/Documents/pokemon_images/images/images/'    # 这里分新旧是为了 将图片中少数jpg格式转成png格式
root_dir = '/home/huangpan/Documents/pokemon_images/images/'
path_list = os.listdir(old_root_dir)
path_list.sort()

for file in path_list:                          # 这里的目的是 将jpg格式的图片转为png格式的图片。通过分割，根据后缀判断文件格式
    # print('file:', file)
    filename_ext = os.path.splitext(file)[1]

    if filename_ext == '.jpg':
        # print('file_jpg:', file)
        im = Image.open(old_root_dir + file)
        im = im.convert('RGBA')
        # print('im.mode:', im.mode)
        filename = os.path.splitext(file)[0]     # 去除扩展名
        im.save(root_dir + filename + '.png')

    elif filename_ext == '.png':
        shutil.copyfile(old_root_dir + file, root_dir + file)
        # print('file_png:', file)
        # im = Image.open(old_root_dir + file)
        # im = im.convert('RGB')
        # print('im.mode:', im.mode)
        # im.save(root_dir + file + '.png')

"""
class_sample = []
label = landmarks_frame['Type1']
print(label, type(label))

for j, i in enumerate(label[:]):
    if i in class_sample:
        pass
    else:
        class_sample.append(i)
        print('{} has add:', i)
print('class_sample:', class_sample)
# landmark = pd.Series(class_sample)
landmarks_Encode = pd.get_dummies(class_sample, sparse=False)
print('landmarks_Encode:', landmarks_Encode)
"""
# label_Onehot = pd.get_dummies(landmarks_frame['Type1'])


def show_landmarks(image, landmarks):
    """
    :param image:
    :param landmarks:  坐标点
    :return:  show image with landmarks
    """
    plt.imshow(image)
    # plt.imshow: input--> array like or PIL image. shape are: (M,N)
    # (M,N,3)- an image with RGB values(0-1 float or 0-255 int)  (M,N,4) return: AxesImage
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(1)


class FaceLanmarksDataset(Dataset):
    """ Face Lanmarks dataset"""
    def __init__(self, csv_file, root_dir, mode, transform=True):            # 加入model，以此将数据集分成train val test等
        """
        :param csv_file (str): Path to the csv file with annotations
        :param root_dir(str): Directory with all the images
        :param transform: optional transform to be applied on a sample
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # self.label_Onehot = label_Onehot                      # 原本是将one-hot标签传进来
        self.transform = transform
        self.name2label = {}

        self.name = self.landmarks_frame['Name']                # 提取csv文件中 Name这列的所有name
        self.label = self.landmarks_frame['Type1']              # 提取csv文件中 Type1这列的所有type
        self.class_sample = []
        self.mode = mode
        # print('len_name:', len(self.name))

        if mode == 'train':                                     # train val test ： 8：1：1
            self.name = self.name[:int(0.9*len(self.name))]
            self.label = self.label[:int(0.9*len(self.label))]
            print('len_train:', len(self.name))
        # elif mode == 'val':
            # self.name = self.name[int(0.9*len(self.name)):int(0.9*len(self.name))]
            # self.label = self.label[int(0.9*len(self.name)):int(0.9*len(self.name))]
            # print('len_val:', len(self.name))
        else:
            self.name = self.name[int(0.9*len(self.name)):]
            self.label = self.label[int(0.9*len(self.name)):]
            print('len_test:', len(self.name))

    def __len__(self):                                        # 返回对应模式下的长度，而不是返回整个样本数据的长度
        return len(self.name)

    def __getitem__(self, idx):
        # print('idx:', idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.png'  # csv文件中只有名字，无png后缀
        landmarks_frame = pd.read_csv('/home/huangpan/Documents/pokemon_images/pokemon.csv')

        label_type = landmarks_frame.iloc[idx, 1]                                          # 该idx行的第一列
        # print('label_type:', type(label_type), label_type)      # <class 'str'> Water
        # if len(self.name2label.keys()) <= 17:
        # self.name2label[label_type] = len(self.name2label.keys())
        # print('len_name2label:', len(self.name2label.keys()))
        # print('name2label:', self.name2label)
        # label_name = landmarks_frame.iloc[idx, 0]
        # self.class_sample = []
        label = landmarks_frame['Type1']                       # <class 'pandas.core.series.Series'>，取该列的所有type类型
        # print('label:',  type(label), label)

        for j, m in enumerate(label[:]):
            if m in self.class_sample:
                pass
            else:
                self.class_sample.append(m)                    # 确认有多少种
                # print('{} has add:', m)
        landmarks = self.class_sample.index(label_type)        # 获取idx对应的type 它在统计出的种类中对应的index
        # print('class_sample:', self.class_sample)
        # print('img_name:', img_name)
        image = io.imread(img_name)                            # 读图片数据。返回Tensor
        image = image.astype(np.float64)

        # image = torch.from_numpy(image)
        # image = image[:, :3, ...]

        # print('image:', type(image), image)
        # landmarks = self.label_Onehot.iloc[idx, 1:].values
        # landmarks = self.name2label.get(label_type)
        sample = {'image': image, 'landmarks': landmarks}     # 返回图片数据和对应的lable

        # print('sample:', sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

"""
face_dataset = FaceLanmarksDataset(label_Onehot=label_Onehot, csv_file='/home/huangpan/Documents/pokemon_images/pokemon.csv',
                                   root_dir='D:/Firefox_Download/pokemon_images/images/')
print('face_dataset:', face_dataset)

fig = plt.figure()
for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break
"""


class Rescale(object):
    """ Rescale the image in a sample to a given size
    Args:
        output_size(tuple or int): Desired output size.If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.

    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))    # 同时验证多种类型. (x, x)和 x
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # print('image:', image.shape)
        h, w = image.shape[:2]
        # print('h:', h, 'w:', w)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size  # outsize为基准缩放图片，以小边为基准并保存长宽比

            else:
                new_h, new_w = self.output_size , self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        # print('Type of new_h:', type(new_w))
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))       # 第一个参数，输入的tensor数据， 第二个参数： 新宽高
        # print('img_Rescale:', img.shape, type(img))
        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """ Crop randomly the image in a sample.

    Args:
        output_size(tuple, int): Desired output size.If int, square crop is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        # print('h, w:', h, w)
        new_h, new_w = self.output_size
        # print('new_h:', new_h)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        # print('image_RandomCrop:', image.shape, image)
        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """ Convert ndarray in sample to Tensor."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # print('ToTensor')
        # print('image_type:', type(image), image.shape)
        # swap color axis because numpy image: H W C  torch image:C H W
        image = image.transpose((2, 0, 1))
        # print('img_ToTensor:', image, image.shape)
        image = torch.from_numpy(image)[:3, :, :]                       # png图片有四个通道RGBA 取前三个通道
        # print('img_new:', image.shape, image)
        # return {'image': torch.from_numpy(image), 'landmarks': landmarks}
        return {'image': image, 'landmarks': landmarks}


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):

        # 并行化模块，降低计算量
        # print('Inception_before_x:', x.shape)     # torch.Size([4, 832, 56, 56])
        y1 = self.b1(x)                           # torch.Size([4, 384, 56, 56])
        # print('Inception_b1_out:', y1.shape)
        y2 = self.b2(x)                           # torch.Size([4, 384, 56, 56])
        # print('Inception_b2_out:', y2.shape)
        y3 = self.b3(x)                           # torch.Size([4, 128, 56, 56])
        # print('Inception_b3_out:', y3.shape)
        y4 = self.b4(x)                           # torch.Size([4, 128, 56, 56])
        # print('Inception_b4_out:', y4.shape)
        return torch.cat([y1, y2, y3, y4], 1)     # 拼接 dim=1 竖着拼 0：横着拼 torch.Size([1, 256, 32, 32])


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

# 网络输入层1层
        self.pre_layers = nn. Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )


# 中间网络层：a3:2层（Inception中为2层横向叠加，增加了宽度）*9 = 18
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)          # 4---->2

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

# 最后一层linear: 共20层  average pooling（平均池化）来代替全连接层，该想法来自NIN（Network in Network），
# 事实证明这样可以将准确率提高0.6%。但是，实际在最后还是加了一个全连接层，主要是为了方便对输出进行灵活调整；
        self.avgpool = nn.AvgPool2d(9, stride=1)
        self.linear = nn.Linear(1024, 18)

    def forward(self, x):
        # print('GoogLeNet_before_x:', x.shape)    # torch.Size([4, 3, 224, 224])

        out = self.pre_layers(x)
        # print('GoogLeNet_pre_layer_out:', out.shape)    # torch.Size([4, 192, 32, 32])
        out = self.a3(out)                              # torch.Size([4, 256, 224, 224])
        # print('GoogLeNet_a3_out:', out.shape)
        out = self.b3(out)                              # torch.Size([4, 480, 224, 224])
        # print('GoogLeNet_b3_out:', out.shape)
        out = self.maxpool(out)                         # torch.Size([4, 480, 112, 112])
        # print('GoogLeNet_maxpool_out:', out.shape)
        out = self.a4(out)                              # torch.Size([4, 512, 50, 50])
        # print('GoogLeNet_a4_out:', out.shape)
        out = self.b4(out)                              # torch.Size([4, 512, 50, 50])
        # print('GoogLeNet_b4_out:', out.shape)
        out = self.c4(out)                              # torch.Size([4, 512, 50, 50])
        # print('GoogLeNet_c4_out:', out.shape)
        out = self.d4(out)                              # torch.Size([4, 528, 50, 50])
        # print('GoogLeNet_d4_out:', out.shape)
        out = self.e4(out)                              # torch.Size([4, 832, 50, 50])
        # print('GoogLeNet_e4_out:', out.shape)
        out = self.maxpool(out)                         # torch.Size([4, 832, 25, 25])
        # print('GoogLeNet_maxpool_out:', out.shape)
        out = self.a5(out)                              # torch.Size([4, 832, 25, 25])
        # print('GoogLeNet_a5_out:', out.shape)
        out = self.b5(out)                              # torch.Size([4, 1024, 35, 35])
        # print('GoogLeNet_b5_out:', out.shape)
        out = self.avgpool(out)                         # torch.Size([4, 1024, 1, 1])
        # print('GoogLeNet_avgpool_out:', out.shape)
        out = out.view(out.size(0), -1)                 # torch.Size([4, 1024])
        # print('GoogLeNet_view_out:', out.shape)
        out = self.linear(out)                          # torch.Size([4, 18])
        # print('GoogLeNet_out:', out.shape)
        return out


if __name__ == "__main__":
    import visdom
    import time
    from torchvision.transforms import ToPILImage
    import torchvision as tv
    # viz = visdom.Visdom()
    show = ToPILImage()
    # scale = Rescale(256)
    # crop = RandomCrop(224)
    # compose = transforms.Compose([Rescale(256), RandomCrop(224)])

    # Apply each of the above transforms on sample

    train_db = FaceLanmarksDataset(csv_file='/home/huangpan/Documents/pokemon_images/pokemon.csv',
                                            root_dir='/home/huangpan/Documents/pokemon_images/images/',
                                            mode='train',
                                            transform=transforms.Compose([Rescale(200), RandomCrop(140),
                                                                          ToTensor(), ]))
    # val_db = FaceLanmarksDataset(csv_file='/home/huangpan/Documents/pokemon_images/pokemon.csv',
                                          # root_dir='/home/huangpan/Documents/pokemon_images/images/',
                                          # mode='val',
                                          # transform=transforms.Compose([Rescale(150), RandomCrop(100),
                                                                        # ToTensor(), ]))
    test_db = FaceLanmarksDataset(csv_file='/home/huangpan/Documents/pokemon_images/pokemon.csv',
                                           root_dir='/home/huangpan/Documents/pokemon_images/images/',
                                           mode='test',
                                           transform=transforms.Compose([Rescale(200), RandomCrop(140),
                                                                         ToTensor(), ]))
    train_loader = DataLoader(train_db, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    # val_loader = DataLoader(val_db, batch_size=4,  num_workers=0, drop_last=True)
    test_loader = DataLoader(test_db, batch_size=4, num_workers=0, drop_last=True)

    # print('len_train_db:', len(train_db), 'len_test_db:', len(test_db))

    net = GoogLeNet().cuda()
    print('net:', net)                                # 打印网络结构

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        runing_loss = 0.
        for i, data in enumerate(train_loader):
            print('i:', '{}'.format(i))
            image, landmarks = data['image'], data['landmarks']
            # print('image:', image.size(), 'landmarks:', landmarks)
            # image = image[:, :3, :, :]

            # print('image_size:', image.size(), type(image))
            image = image.float()                               # 根据bug提示，转换类型
            image = image.cuda()                                # 利用GPU
            landmarks = landmarks.long()
            # landmarks = landmarks.float()
            landmarks = landmarks.cuda()
            image, landmarks = Variable(image, requires_grad=True), Variable(landmarks)     # 默认是false， 这里要求梯度
            # print('image_type:', type(image))
            optimizer.zero_grad()

            outputs = net(image)
            # print('outputs:', outputs)
            loss = criterion(outputs, landmarks)
            print('loss:', loss)
            # print(net.pre_layers.weight)
            loss.backward()
            optimizer.step()

            runing_loss += loss.item()
            # print('runing_loss:', runing_loss)
            if i % 100 == 0:                                                         # 每隔100打印一次，统计100次的平均损失
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, runing_loss/100))
                runing_loss = 0.0
    print('Finish Training')

    # show real image and related label
    # with torchsnooper.snoop():
    dataiter = iter(test_loader)                                                    # 展示测试集中batchsize的图片标签
    sample = dataiter.next()
    image, landmarks = sample['image'], sample['landmarks']
    # print('image:', image.shape,  image.dtype, image.type(), image, 'landmarks:', landmarks)
    print('real label:', ' '.join('%08s' % test_db.class_sample[landmarks[i]] for i in range(4)))

    """
    for i in range(4):
        a = image[i]
        b = transforms.ToPILImage()(a.float())                                  
        b.show()
        """

    # predicted the label of image
    image = image.float()
    image = image.cuda()
    # landmarks = landmarks.long()
    # landmarks = landmarks.float()
    # landmarks = landmarks.cuda()
    image = Variable(image, requires_grad=False)

    outputs = net(image)                                          # 将上面测试集的数据喂给网络，预测是什么。
    # print('outputs_type:', type(outputs), outputs.dtype)
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted the label:', ' '.join('%08s' % test_db.class_sample[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    for data in test_loader:
        images, landmarks = data['image'], data['landmarks']
        image = image.float()
        image = image.cuda()

        landmarks = landmarks.long()
        landmarks = landmarks.cuda()
        image = Variable(image, requires_grad=False)

        outputs = net(image)
        # print('outputs_type2:', type(outputs), outputs.dtype)
        _, predicted = torch.max(outputs, 1)
        total += landmarks.size(0)
        correct += (predicted == landmarks).sum()
    print('correct:', correct)
    print('80 pic accuracy:, %d/%d ,%d %%' % (correct, total, 100 * correct / total))



















