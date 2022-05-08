import cv2
import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import get_config


def denorm(tensor: torch.Tensor):
    tensor = tensor * tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = tensor + tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    return tensor.clamp(0, 1)


def text_to_image(image, y_true, y_pred):
    y_true = f"{y_true:0.2f}"
    y_pred = f"{y_pred:0.2f}"

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("util/arial.ttf", 30)
    green = (54, 179, 9)
    red = (173, 17, 23)
    draw.text(text=y_true, xy=(2, 10), font=font, fill=green)
    draw.text(text=y_pred, xy=(150, 10), font=font, fill=red)

    image = np.array(img)
    image = cv2.resize(image, (256, 256))
    return image


def images_to_image(images):
    sz1 = 2
    sz0 = 2
    images = images[:sz1 * sz0]

    X = np.zeros((sz0 * images.shape[1], sz1 * images.shape[2], 3))
    for i in range(sz0):
        for j in range(sz1):
            if len(images) <= i * sz1 + j:
                continue
            crop = images[i * sz1 + j]
            X[i * images.shape[1]: (i + 1) * images.shape[1], j * images.shape[2]: (j + 1) * images.shape[2]] = crop
    return X


def visualize_images(images, y_true, y_pred):
    images = images.detach()[:, :3]
    images = denorm(images)
    images = images.cpu().numpy()
    images = images.transpose((0, 2, 3, 1))
    y_true = y_true.cpu().detach().numpy().flatten()
    y_pred = y_pred.cpu().detach().numpy().flatten()

    patch = []
    for i, image in enumerate(images):
        image = np.array(image * 255, dtype=np.uint8)
        patch.append(text_to_image(image, y_true[i], y_pred[i]))
    image = np.array(images_to_image(np.array(patch)), dtype=np.uint8)
    image = torch.from_numpy(image.transpose(2, 0, 1))
    return image


def save_model(model, optimizer, scheduler, scaler, epoch, config):
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'scaler': scaler.state_dict(),
        },
        '{}/Epoch-{}.pt'.format(config.LOG.SAVED_MODELS, epoch)
    )
    print("Model save: ", '{}/Epoch {}.pt'.format(config.LOG.SAVED_MODELS, epoch))


def parse_option():
    parser = argparse.ArgumentParser("SRDM")
    parser.add_argument('--model-name', type=str, default="resnet", help='Model to train, possible options - resnet, mobilenet')
    parser.add_argument('--version', type=str, default="", help="train version, template -- frames: _, loss: _, emb: _")
    parser.add_argument("--ce", type=int, default=1, help="Include cross-entropy loss? 0 -- False, 1 -- True")
    parser.add_argument("--trp", type=int, default=1, help="Include triplet loss? 0 -- False, 1 -- True")
    parser.add_argument("--std", type=int, default=1, help="Include variance loss? 0 -- False, 1 -- True")
    parser.add_argument('--pretrained', type=str, default="", help='pretrained weight from checkpoint')
    parser.add_argument('--embedding-size', type=int, default=64, help="embedding size")
    parser.add_argument('--n_frames', type=int, default=2, help="the number of frames")

    parser.add_argument('--batch-size', type=int, default=32, help="batch size for GPU")
    parser.add_argument('--accumulation-steps', type=int, default=1, help="gradient accumulation steps")

    parser.add_argument('--test-data-path', type=str, default="", help='path to test dataset')

    parser.add_argument('--num_workers', type=int, default=2, help="the number of workers")
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config
