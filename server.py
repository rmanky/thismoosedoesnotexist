from io import BytesIO

import torch
from flask import Flask, jsonify, send_file
from torchvision.transforms import transforms

from models import Generator

app = Flask(__name__)


@app.route('/')
def predict():
    image_size = 64
    channels_img = 3
    channels_noise = 128
    features_g = 64
    fixed_noise = torch.randn(image_size, channels_noise, 1, 1)
    netG = Generator(channels_noise, channels_img, features_g)
    netG.load_state_dict(torch.load('g.pth'))
    fake = netG(fixed_noise)
    newFake = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage(),
        transforms.Resize(512)
    ])(fake[0])
    print(newFake)
    return serve_pil_image(newFake)


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
