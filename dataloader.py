import os
import cv2
import json
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from torch.utils.data import DataLoader, Dataset


def read_cv_image(filename):
    image = cv2.imread(filename)
    assert image is not None, f"image: {filename}  is None"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


transform_normalize = albu.Compose([
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    albu.pytorch.transforms.ToTensorV2(),
], p=1)


def build_dataloader(config):
    path_to_video_dataset = config.DATA.DATA_PATH_TRAIN
    gt_paths = ['Original/GT', 'Compressed/GT']
    sr_paths = [f"Original/{dataset}" for dataset in config.DATA.SR_METHODS_TRAIN] + \
               [f"Compressed/{dataset}" for dataset in config.DATA.SR_METHODS_TRAIN]

    additional_targets = {f"frame_{i}": "image" for i in range(1, config.DATA.N_FRAMES)}

    datasetTrain = DataLoader(
        CustomVideoFramesDatasetLoaderTriplet(gt_paths, sr_paths, path_to_video_dataset, n_frames=config.DATA.N_FRAMES,
                                              transforms=albu.Compose([
                                                  albu.RandomCrop(config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE, p=1),
                                                  CoarseDropout(max_holes=2,
                                                                max_height=100,
                                                                max_width=100),
                                                  transform_normalize,
                                              ], p=1, additional_targets=additional_targets)),
        batch_size=config.TRAIN.BATCH_SIZE, shuffle=True, num_workers=config.DATA.NUM_WORKERS, pin_memory=True)

    # Without compressing
    folder = f'{config.DATA.DATA_PATH_TEST}/Original'
    datasets_original_video = ["GT"] + config.DATA.SR_METHODS_TEST
    datasetValidVideoOriginal = dict()
    for dataset in datasets_original_video:
        datasetValidVideoOriginal[dataset] = DataLoader(
            CustomVideoFramesDatasetLoader([os.path.join(folder, dataset)], originals=[bool("GT" in dataset)],
                                           transforms=albu.Compose([
                                               transform_normalize
                                           ], p=1, additional_targets=additional_targets),
                                           n_frames=config.DATA.N_FRAMES),
            batch_size=config.TEST.BATCH_SIZE, shuffle=False, num_workers=config.DATA.NUM_WORKERS)

    # With video compressing
    folder = f'{config.DATA.DATA_PATH_TEST}/Compressed'
    datasets_compressed_video = ["GT"] + config.DATA.SR_METHODS_TEST
    datasetValidVideoCompressed = dict()
    for dataset in datasets_compressed_video:
        datasetValidVideoCompressed[dataset] = DataLoader(
            CustomVideoFramesDatasetLoader([os.path.join(folder, dataset)], originals=[bool("GT" in dataset)],
                                           transforms=albu.Compose([
                                               transform_normalize
                                           ], p=1, additional_targets=additional_targets),
                                           n_frames=config.DATA.N_FRAMES),
            batch_size=config.TEST.BATCH_SIZE, shuffle=False, num_workers=config.DATA.NUM_WORKERS)

    return datasetTrain, datasetValidVideoOriginal, datasetValidVideoCompressed


class CustomVideoFramesDatasetLoader(Dataset):
    def __init__(self, folders, originals, transforms=None, n_frames=1, shuffle=False, random_file=True):
        list_of_files = []
        labels = []
        dtype = int
        for i, folder in enumerate(folders):
            original = originals[i]
            tmp_list = [os.path.join(folder, file) for file in sorted(os.listdir(folder))]
            list_of_files += tmp_list

            if original:
                labels += np.zeros(len(tmp_list), dtype=dtype).tolist()
            else:
                labels += np.ones(len(tmp_list), dtype=dtype).tolist()

        if shuffle:
            dataset_size = len(list_of_files)
            idx = np.random.permutation(dataset_size).astype(int)
            list_of_files, labels = np.array(list_of_files)[idx], np.array(labels)[idx]

        self.list_of_files = list_of_files
        self.labels = np.array(labels, dtype=dtype)
        self.transforms = transforms
        self.random_file = random_file
        self.n_frames = n_frames

    def __len__(self):
        return len(self.list_of_files)

    def _get_random_frames(self, videoname):
        n_frames = self.n_frames
        frame_names = sorted(os.listdir(str(videoname)))
        index = np.random.randint(0, max(1, len(frame_names) - n_frames + 1))
        if len(frame_names) == 0:
            print(f"Error - empty video: {videoname}")

        frames = []
        for i_ in range(index, index + n_frames):
            i = min(i_, len(frame_names) - 1)

            filename_frame = os.path.join(videoname, frame_names[i])
            frame = read_cv_image(filename_frame)
            frames.append(frame)

        args = {'image': frames[0]}
        for i in range(1, n_frames):
            args[f"frame_{i}"] = frames[i]

        if self.transforms is not None:
            result = self.transforms(**args)
            frames = [result["image"]]
            for i in range(1, n_frames):
                frames.append(result[f"frame_{i}"])

        return np.concatenate(frames, axis=0)

    def __getitem__(self, idx):
        videoname = self.list_of_files[idx]
        label = self.labels[idx]
        frames = self._get_random_frames(videoname)
        return frames, label


class CustomVideoFramesDatasetLoaderTriplet(Dataset):
    def __init__(self, gt_paths, sr_paths, path_to_video_dataset, n_frames=2, transforms=None):
        video_to_map = dict()
        for gt_dataset in gt_paths:
            for video in os.listdir(f"{path_to_video_dataset}/{gt_dataset}"):
                if video not in video_to_map:
                    video_to_map[video] = {'sr': [], 'gt': []}
                video_to_map[video]['gt'].append(f"{path_to_video_dataset}/{gt_dataset}/{video}")

        for sr_dataset in sr_paths:
            for video in os.listdir(f"{path_to_video_dataset}/{sr_dataset}"):
                if video not in video_to_map:
                    video_to_map[video] = {'sr': [], 'gt': []}
                video_to_map[video]['sr'].append(f"{path_to_video_dataset}/{sr_dataset}/{video}")

        self.ind_to_video = {i: video for i, video in enumerate(video_to_map)}
        self.video_to_map = video_to_map
        self.transforms = transforms
        self.n_frames = n_frames
        self._len = len(self.video_to_map)
        with open("log.json", "w") as fout:
            json.dump(video_to_map, fout)

    def __len__(self):
        return self._len

    def _get_random_frames(self, videoname):
        n_frames = self.n_frames
        frame_names = sorted(os.listdir(str(videoname)))
        index = np.random.randint(0, max(1, len(frame_names) - n_frames + 1))
        if len(frame_names) == 0:
            print(f"Error - empty video: {videoname}")

        frames = []
        for i_ in range(index, index + n_frames):
            i = min(i_, len(frame_names) - 1)

            filename_frame = os.path.join(videoname, frame_names[i])
            frame = read_cv_image(filename_frame)
            frames.append(frame)

        args = {'image': frames[0]}
        for i in range(1, n_frames):
            args[f"frame_{i}"] = frames[i]

        if self.transforms is not None:
            result = self.transforms(**args)
            frames = [result["image"]]
            for i in range(1, n_frames):
                frames.append(result[f"frame_{i}"])

        return np.concatenate(frames, axis=0)

    def __getitem__(self, idx):

        videoname = self.ind_to_video[idx]
        gt_list = self.video_to_map[videoname]['gt']
        ind = np.random.randint(0, len(gt_list))
        s = f"gt: {idx}, {ind}"
        gt_video = gt_list[ind]
        anchor_frames = self._get_random_frames(gt_video)

        sr_list = self.video_to_map[videoname]['sr']
        ind = np.random.randint(0, len(sr_list))
        s += f"sr: {ind}"
        sr_video = sr_list[ind]
        neg_frames = self._get_random_frames(sr_video)

        pos_idx = np.random.randint(0, self._len)
        videoname = self.ind_to_video[pos_idx]
        pos_list = self.video_to_map[videoname]['gt']
        ind = np.random.randint(0, len(pos_list))
        s += f"pos: {pos_idx}, {ind}"
        pos_video = pos_list[ind]
        pos_frames = self._get_random_frames(pos_video)

        return anchor_frames, pos_frames, neg_frames
