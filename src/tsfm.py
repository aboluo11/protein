from lightai.core import *

mean = np.array([20.50361 , 13.947072, 13.408824, 21.106398]).reshape((-1, 1, 1))
std = np.array([38.12811 , 39.742226, 28.598948, 38.173912]).reshape((-1, 1, 1))

def get_img(row, sz, train):
    colors = ['yellow', 'red', 'green', 'blue']
    channels = []
    for color in colors:
        name = row['Id'] + f'_{color}.png'
        if train:
            img_path = f'inputs/{sz}_train/{name}'
        else:
            img_path = f'inputs/{sz}_test/{name}'
        channel = cv2.imread(img_path, -1)
        channels.append(channel)
    img = np.stack(channels)
    img = img.astype(np.float32)
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
            img = self.img_tsfm(img)
        return img, target

class TestTsfm:
    def __init__(self, sz):
        self.sz = sz

    def __call__(self, row):
        img = get_img(row, self.sz, False)
        return img


class ComposeTsfms:
    def __init__(self, tsfms):
        self.tsfms = tsfms
    def __call__(self, img):
        for tsfm in self.tsfms:
            img = tsfm(img)
        return img


class LR_Flip:
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if np.random.rand() < self.p:
            img = img[:,:,::-1].copy()
        return img


class VerticalFlip:
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if np.random.rand() < self.p:
            img = img[:,::-1,:].copy()
        return img