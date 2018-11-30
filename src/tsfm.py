from lightai.core import *

# mean = np.array([20.50361, 13.947072, 13.408824,
#                  21.106398]).reshape((-1, 1, 1))
# std = np.array([38.12811, 39.742226, 28.598948, 38.173912]).reshape((-1, 1, 1))
# mean = np.array([31.6239, 24.2544, 14.9083]).reshape((-1, 1, 1))
# std = np.array([46.4143, 36.4069, 41.0179]).reshape((-1, 1, 1))
mean = np.array([22.4812, 13.4036, 14.9883])
std = np.array([38.5135, 27.0687, 40.9111])

def get_img(row, sz, train):
    colors = ['red', 'green', 'blue']
    channels = []
    for color in colors:
        name = row['Id'] + f'_{color}.png'
        if train:
            img_path = f'inputs/{sz}_train/{name}'
        else:
            img_path = f'inputs/{sz}_test/{name}'
        channel = cv2.imread(img_path, -1)
        channels.append(channel)
    img = np.stack(channels, axis=-1)
    # img[:, :, 1] += img[:, :, 0]//2
    # img[:, :, 2] += img[:, :, 0]//2
    # img = img[:, :, 1:]
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
        return img
