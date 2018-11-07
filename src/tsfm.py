from lightai.core import *

def get_img(row, color):
    name = row['Id'] + f'_{color}.png'
    img = cv2.imread(f'inputs/train/{name}')
    return img

def get_target(row):
    targets = row['Target'].split()
    targets = [int(t) for t in targets]
    res = np.zeros(28, dtype=np.float32)
    res[targets] = 1
    return res

def tsfm(row):
    green = get_img(row, 'green')
    target = get_target(row)
    return green, target