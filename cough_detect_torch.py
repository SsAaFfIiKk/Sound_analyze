import os
import torch
import librosa
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
from math import ceil
from torchvision import transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3),
            padding=1)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=1)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.batch1 = nn.BatchNorm2d(num_features=16)
        self.batch2 = nn.BatchNorm2d(num_features=32)
        self.batch3 = nn.BatchNorm2d(num_features=64)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.act = nn.ReLU()

        self.dropout = nn.Dropout(inplace=True)

        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        h = self.conv1(x)
        h = self.batch1(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv2(h)
        h = self.batch2(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv3(h)
        h = self.batch3(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv4(h)
        h = self.batch3(h)
        h = self.act(h)

        h = self.conv5(h)
        h = self.batch3(h)
        h = self.act(h)
        h = self.pool(h)

        h = h.view(h.size(0), -1)

        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc3(h)
        return h


def chunkizer(audio, sr, chunk_length):
    num_chunks = ceil(librosa.get_duration(audio, sr=sr) / chunk_length)
    chunks = []
    for i in range(num_chunks):
        chunks.append(audio[i*chunk_length*sr:(i+1)*chunk_length*sr])
    return chunks


path_to_audio = "F:/Python/Data/demo/audio_only_ivan.wav"

model = torch.load("wight.pth").cpu()
model.eval()
get_label = {0: "speech", 1: "cough"}
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.0), (1.0))])

y, sr = librosa.load(path_to_audio)
chunks = chunkizer(y, sr, 10)
cmap = plt.get_cmap('inferno')

ansers = []

for i, chunk in enumerate(chunks):
    plt.specgram(chunk, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
    plt.axis('off')
    plt.savefig(f'{i}.png')
    plt.clf()

    img = Image.open(f'{i}.png')
    tensor = test_transform(img)
    tensor = torch.unsqueeze(tensor, 0)

    with torch.no_grad():
        result = model(tensor)
        label = result.argmax(dim=1)
        anser = get_label[label.sum().item()]
        ansers.append(anser)

print(ansers.count("cough"))
files = os.listdir()
for file in files:
    if file.endswith(".png"):
        os.remove(file)
