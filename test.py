import os
import json
import torch
from tqdm import tqdm
import numpy as np
from torch.nn.functional import softmax

from dataloader import build_test_dataloader
from utils import parse_option
from builders import build_model


def test_criterion(probas, config):
    quantile = np.quantile(probas, config.TEST.QUANTILE)
    return quantile > 0.5


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = build_model(config).to(device)

    result = dict()
    for video in os.listdir(config.TEST.DATA_PATH):
        result[video] = test_video(model, video, config)
        print(f"Video {video}: {'Fake' if result[video] else 'Real'}")

    file_name_result = f"{config.TEST.RESULT_PATH}.json"
    with open(file_name_result, "w") as fout:
        json.dump(result, fout)
    return result


def test_video(model, video, config):
    model.eval()
    dataloader = build_test_dataloader(f"{config.TEST.DATA_PATH}/{video}", config)

    result = []
    tqdm_ = tqdm(dataloader, desc=f"Process video: {video}")
    with torch.no_grad():
        for frames in tqdm_:
            frames = frames.cuda(non_blocking=True)
            pred = softmax(model(frames).detach().cpu(), dim=1)[0, 1].item()
            result.append(pred)
    return test_criterion(result, config)


if __name__ == "__main__":
    config = parse_option()
    print("Config:"
          "\n-------------------------------------------------------------\n",
          config,
          "\n-------------------------------------------------------------\n", sep="")
    main(config)
