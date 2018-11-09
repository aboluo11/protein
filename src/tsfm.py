from lightai.core import *

def get_img(row, color, sz, train):
    name = row['Id'] + f'_{color}.png'
    if train:
        img_path = f'inputs/{sz}_train/{name}'
    else:
        img_path = f'inputs/{sz}_test/{name}'
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    return img

def get_target(row):
    targets = row['Target'].split()
    targets = [int(t) for t in targets]
    res = np.zeros(28, dtype=np.float32)
    res[targets] = 1
    return res

class Tsfm:
    def __init__(self, sz):
        self.sz = sz

    def __call__(self, row):
        green = get_img(row, 'green', self.sz, True)
        target = get_target(row)
        return green, target

class TestTsfm:
    def __init__(self, sz):
        self.sz = sz

    def __call__(self, row):
        green = get_img(row, 'green', self.sz, False)
        return green