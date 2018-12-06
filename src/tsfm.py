from lightai.core import *


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


def get_target(row):
    targets = row['Target'].split()
    targets = [int(t) for t in targets]
    res = np.zeros(28, dtype=np.float32)
    res[targets] = 1
    return res


class Tsfm:
    def __init__(self, sz, img_tsfm=None):
        self.sz = sz
        self.img_tsfm = img_tsfm

    def __call__(self, row):
        img = get_img(row, self.sz, True)
        target = get_target(row)
        if self.img_tsfm:
            img = self.img_tsfm(image=img)['image']
        return img, target


class TestTsfm:
    def __init__(self, sz):
        self.sz = sz

    def __call__(self, row):
        img = get_img(row, self.sz, False)
        imgs = [img]
        imgs.append(img[::-1].copy())
        imgs.append(img[:, ::-1].copy())
        imgs.append(img[::-1, ::-1].copy())
        return imgs
