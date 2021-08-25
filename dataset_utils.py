from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch import Tensor
from torch import functional as F
import math


# this class is an extension of Pytorchs auto augment that uses Albumentation as a back end instead
class AlbumentationsAutoAugment(transforms.autoaugment.AutoAugment):
    def forward(self, image, force_apply=True):
        fill = self.fill
        if isinstance(image, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(image)
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.transforms))

        for i, (op_name, p, magnitude_id) in enumerate(self.transforms[transform_id]):
            if probs[i] <= p:
                magnitudes, signed = self._get_op_meta(op_name)
                magnitude = float(magnitudes[magnitude_id].item()) \
                    if magnitudes is not None and magnitude_id is not None else 0.0
                if signed is not None and signed and signs[i] == 0:
                    magnitude *= -1.0

                if op_name == "ShearX":
                    transform = A.Affine(shear={'x': math.degrees(magnitude)}, cval=fill)
                    return transform(image=image)
                elif op_name == "ShearY":
                    transform = A.Affine(shear={'y': math.degrees(magnitude)}, cval=fill)
                    return transform(image=image)
                elif op_name == "TranslateX":
                    transform = A.Affine(translate_px={'x': int(image.shape[0] * magnitude)}, cval=fill)
                    return transform(image=image)
                elif op_name == "TranslateY":
                    transform = A.Affine(translate_px={'y': int(image.shape[0] * magnitude)}, cval=fill)
                    return transform(image=image)
                elif op_name == "Rotate":
                    image = A.rotate(image, magnitude)
                elif op_name == "Brightness":
                    image = A.adjust_brightness_torchvision(image, 1.0 + magnitude)
                elif op_name == "Color":
                    image = A.adjust_saturation_torchvision(image, 1.0 + magnitude)
                elif op_name == "Contrast":
                    image = A.adjust_contrast_torchvision(image, 1.0 + magnitude)
                elif op_name == "Sharpness":
                    image = A.adjust_contrast_torchvision(image, 1.0 + magnitude)
                elif op_name == "Posterize":
                    image = A.posterize(image, int(magnitude))
                elif op_name == "Solarize":
                    image = A.solarize(image, threshold=magnitude)
                elif op_name == "AutoContrast":
                    image = A.brightness_contrast_adjust(image)
                elif op_name == "Equalize":
                    image = A.equalize(image)
                elif op_name == "Invert":
                    image = A.invert(image)
                else:
                    raise ValueError("The provided operator {} is not recognized.".format(op_name))

        return {'image': image}

    def _to_dict(self):
        return {'__class__fullname__': 'AlbumentationsAutoAugment', 'policy': self.policy}


# from https://github.com/pytorch/vision/blob/master/references/classification/presets.py
class ClassificationPresetTrain:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), hflip_prob=0.5,
                 auto_augment_policy=None, random_erase_prob=0.0, use_albumentations=True):
        self.use_albumentations = use_albumentations

        if use_albumentations:
            trans = [A.RandomResizedCrop(crop_size, crop_size)]
            if hflip_prob > 0:
                trans.append(A.HorizontalFlip(p=hflip_prob))
            if auto_augment_policy is not None:
                aa_policy = transforms.autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(AlbumentationsAutoAugment(policy=aa_policy))
            if random_erase_prob > 0:
                trans.append(A.CoarseDropout(p=random_erase_prob))

            trans.extend([
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

            self.transforms = A.Compose(trans)
        else:
            trans = [transforms.RandomResizedCrop(crop_size)]
            if hflip_prob > 0:
                trans.append(transforms.RandomHorizontalFlip(hflip_prob))
            if auto_augment_policy is not None:
                aa_policy = transforms.autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(transforms.autoaugment.AutoAugment(policy=aa_policy))
            trans.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            if random_erase_prob > 0:
                trans.append(transforms.RandomErasing(p=random_erase_prob))

            self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        if self.use_albumentations:
            return self.transforms(image=img)['image']
        else:
            return self.transforms(img)


class ClassificationPresetEval:
    def __init__(self, crop_size, resize_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 use_albumentations=True):
        self.use_albumentations = use_albumentations

        if use_albumentations:
            self.transforms = A.Compose([
                A.Resize(resize_size, resize_size),
                A.CenterCrop(crop_size, crop_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __call__(self, img):
        if self.use_albumentations:
            return self.transforms(image=img)['image']
        else:
            return self.transforms(img)


# this class uses a separate cuda stream to pre-load the batch onto GPU memory, 2% increase in perf
# inspired from :
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py
# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
class CudaPrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = False
        data = None

        for next_data in self.loader:
            with torch.cuda.stream(stream):
                next_data[0] = next_data[0].to(self.device, non_blocking=True)
                next_data[1] = next_data[1].to(self.device, non_blocking=True)

            if first:
                yield data
            else:
                first = True

            torch.cuda.current_stream().wait_stream(stream)
            data = next_data

        yield data

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)
        ~1.6% performance increase.
    """
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


class PreProcessedDataset(Dataset):
    def __init__(self, dataset):
        self.data = []
        for data in dataset:
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
