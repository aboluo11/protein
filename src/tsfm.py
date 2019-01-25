from lightai.core import *
from lightai.torch_core import *
from albumentations import *


def get_img(row, sz, train):
    colors = ['red', 'green', 'blue', 'yellow']
    channels = []
    for color in colors:
        name = row['Id'] + f'_{color}.png'
        if train:
            # img_path = f'inputs/{sz}_full_external/{name}'
            img_path = f'inputs/{sz}_train/{name}'
        else:
            img_path = f'inputs/{sz}_test/{name}'
        channel = cv2.imread(img_path, -1)
        channels.append(channel)
    img = np.stack(channels, axis=-1)
    return img


def get_1728_img(row, sz, train):
    colors = ['red', 'green', 'blue', 'yellow']
    channels = []
    for color in colors:
        if train:
            name = row['Id'] + f'_{color}.jpg'
            img_path = f'inputs/1728_train/{name}'
        else:
            name = row['Id'] + f'_{color}.jpg'
            img_path = f'inputs/1024_test/{name}'
        channel = cv2.imread(img_path, -1)
        channels.append(channel)
    img = np.stack(channels, axis=-1)
    if train:
        img = cv2.resize(img, (sz, sz))
    return img


def get_target(row):
    targets = row['Target'].split()
    targets = [int(t) for t in targets]
    res = np.zeros(28, dtype=np.float32)
    res[targets] = 1
    return res


class Tsfm:
    def __init__(self, sz, fair_img_tsfm=None, weighted_img_tsfm=None):
        self.sz = sz
        self.fair_img_tsfm = fair_img_tsfm
        self.weighted_img_tsfm = weighted_img_tsfm

    def __call__(self, row):
        img = get_1728_img(row, self.sz, True)
        target = get_target(row)
        if self.fair_img_tsfm:
            img = self.fair_img_tsfm(image=img)['image']
        if self.weighted_img_tsfm:
            weight = row['weight']
            p = weight*0.25 + 0.5
            if np.random.rand() < p:
                img = self.weighted_img_tsfm(image=img)['image']
            # img = self.weighted_img_tsfm(image=img)['image']
        return img, target


class TestTsfm:
    def __init__(self, sz, tta=True):
        self.sz = sz
        self.tta = tta
        self.brightness = RandomBrightnessContrast(
            brightness_limit=(0.1, 0.11), contrast_limit=(-0, 0))

    def __call__(self, row):
        img = get_img(row, self.sz, False)
        if not self.tta:
            return img
        imgs = []
        for transpose in [0, 1]:
            for h_flip in [0, 1]:
                for v_flip in [0, 1]:
                    for brightness in [0, 1]:
                        tta = img
                        if transpose:
                            tta = np.transpose(tta, axes=(1, 0, 2))
                        if h_flip:
                            tta = tta[:, ::-1]
                        if v_flip:
                            tta = tta[::-1]
                        if brightness:
                            tta = self.brightness(image=tta)['image']
                        imgs.append(tta.copy())
        return imgs
