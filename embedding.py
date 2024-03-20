import torch
import torch.nn as nn
import torch.nn.functional as F

#BasicBlock class adapted from torch resnet.py
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BlockGroup(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=2, stride=1):
        super(BlockGroup, self).__init__()
        first = BasicBlock(in_channels,out_channels,stride=stride)
        blocks = [first] + [BasicBlock(out_channels,out_channels,stride=stride) for i in range(n_blocks-1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self,x):
        out = x
        return self.blocks(out)

#embedding dimension is channels_per_group[-1]
class ResnetEmbedder(nn.Module):
    def __init__(self, in_channels, conv1_channels=8, channels_per_group=[16,32,32,64], blocks_per_group=[2,2,2,2], strides=[1,1,1,1]):
        super(ResnetEmbedder, self).__init__()
        assert len(channels_per_group) == len(blocks_per_group), "list dimensions must match"
        assert len(blocks_per_group) == len(strides), "list dimensions must match"
        assert len(channels_per_group) >= 1, "must have at least one layer"

        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        layers = []
        for i, (c,b,s) in enumerate(zip(channels_per_group, blocks_per_group, strides)):
            if i == 0:
                layers.append(BlockGroup(conv1_channels,c,n_blocks=b,stride=s))
            else:
                prev_c = channels_per_group[i-1]
                layers.append(BlockGroup(prev_c,c,n_blocks=b,stride=s))
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        out = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        #average over superpixels to create embedding
        out = torch.mean(out,axis=(2,3))
        return out