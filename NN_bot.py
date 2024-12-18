import os
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import ContentType
import asyncio
import pandas as pd
import torch
from torch import nn, optim
from torchvision import transforms, models
from torch_snippets import *
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary 

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import cv2
from glob import glob
import pandas as pd

sns.set_theme()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DIR = "D:/NIK_CINICHKA/HH/Human Action Recognition/"
TRAIN_DIR=f"{DIR}train"
TEST_DIR=f"{DIR}test"
TRAIN_VAL_DF = "D:/NIK_CINICHKA/HH/Human Action Recognition/Training_set.csv"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ActionClassifier(nn.Module):
    def __init__(self, ntargets):
        super().__init__()
        resnet = models.resnet50(pretrained=True, progress=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(resnet.fc.in_features),
            nn.Dropout(0.2),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, ntargets)
        )
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x


df = pd.read_csv("D:/NIK_CINICHKA/HH/Human Action Recognition/Training_set.csv")
ind2cat = sorted(df['label'].unique().tolist())
cat2ind = {cat: ind for ind, cat in enumerate(ind2cat)}

classifier = ActionClassifier(len(ind2cat)).to(device)
classifier.load_state_dict(torch.load('./saved_model/classifier_weights.pth'))
classifier.eval()


async def predict_class(img_path):
    img = Image.open(img_path).convert('RGB')

    img = transforms.Resize([224, 224])(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.485, 0.456, 0.406), (0.229*255, 0.224*255, 0.225*255))(img)
    img = img.unsqueeze(0).to(device)  

    with torch.no_grad():  
        output = classifier(img)
        predicted_class = output.argmax(1).item() 

    return ind2cat[predicted_class]


API_TOKEN = "7709905603:AAFKwjsQjXzqcJhDMtMAcyilmuO8jtjBtBw"  
bot = Bot(token=API_TOKEN)
dp = Dispatcher()


@dp.message(Command('start'))
async def cmd_start(message: types.Message):
    await message.answer("Hello! Send me a photo and I will classify it.")


@dp.message(lambda message: message.content_type == ContentType.PHOTO)
async def handle_photo(message: Message):
    file_id = message.photo[-1].file_id  
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "user_image.jpg")

    predicted_class = await predict_class("user_image.jpg") 
    await message.answer(f"Думаю здесь: {predicted_class}")

if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot, skip_updates=True))
