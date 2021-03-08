from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

DATA_RAW_DIR = 'img_correct'
DATA_FRAME_DIR = 'img_extract'
CHAR_WIDTH = 25

def func_img4(img4):
    _, thres = cv2.threshold(img4,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT ,(2,1))
    close1 = cv2.morphologyEx(thres,cv2.MORPH_CLOSE,kernel1, iterations=1)
    blur1 = cv2.medianBlur(close1, 3)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    close2 = cv2.morphologyEx(thres,cv2.MORPH_CLOSE,kernel2, iterations=1)
    blur2 = cv2.medianBlur(close2, 3)

    ret = blur1 if np.average(blur1[:, :40]) <= np.average(blur2[:, :40]) else blur2
    return ret

def func_subtract(img_new, img_old):
    img8 = cv2.subtract(img_old, img_new)
    _, thres8 = cv2.threshold(img8,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS ,(3,3))
    opened = cv2.morphologyEx(thres8,cv2.MORPH_OPEN,kernel, iterations=1)
    return cv2.medianBlur(opened, 3)

def img_crop(image, w):
    if w >= 0 and w + 52 <= image.shape[1]:
        return image[:, w:w + 52]
    
    cropped = 255 - np.zeros((52, 52))
    if w < 0:
        cropped[:, -w:52] = image[:, :52 + w]
    else:
        cropped[:, :image.shape[1]-w] = image[:, w:]
    return cropped

def cropping(proced_img, orig_img, first = False):
    blur = cv2.medianBlur(proced_img, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS ,(11,11))
    opened = cv2.morphologyEx(blur, cv2.MORPH_OPEN,kernel, iterations=3)

    # following codes by xmcp
    imwidth = 52 if first else proced_img.shape[1]
    sums_prefix = np.sum(255 - opened, 0)
    for i in range(1, imwidth):
        sums_prefix[i]+=sums_prefix[i-1]
    
    ma = -1
    mapos = -1
    for i in range(CHAR_WIDTH,imwidth):
        v = sums_prefix[i] - sums_prefix[i-CHAR_WIDTH]
        if v>ma:
            ma = v
            mapos = i
    left = mapos - CHAR_WIDTH - (52-CHAR_WIDTH)//2
    return img_crop(orig_img, left)

def gen(img):
    ret = [None]*4
    last_img = None

    mask = np.ones((52, 130), dtype=np.uint8)
    
    for i, index in enumerate((3, 7, 11, 15)):
        img.seek(index)
        img4 = Image.new("RGB", img.size)
        img4.paste(img)
        img4 = np.array(img4)
        img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2GRAY)

        if index == 3:
            new_img = func_img4(img4)
            #Image.fromarray(new_img.astype(np.uint8)).save('step%d.png'%i)
            #plt.imshow(new_img&mask)
            #plt.show()
            crop_img = cropping(new_img, new_img, first=True)
        else:
            new_img = func_subtract(img4, last_img)
            #Image.fromarray(new_img.astype(np.uint8)).save('step%d.png'%i)
            #plt.imshow(255-((255-new_img)&mask))
            #plt.show()
            crop_img = cropping(255-((255-new_img)&mask), new_img)
        
        last_img = img4
        mask &= new_img.astype(np.bool)
        ret[i] = crop_img.astype(np.uint8)

    return ret

def main():
    if not os.path.exists(DATA_FRAME_DIR):
        os.mkdir(DATA_FRAME_DIR)

    for img_name in tqdm(os.listdir(DATA_RAW_DIR)):
        gen(Image.open(os.path.join(DATA_RAW_DIR, img_name)))

if __name__ == "__main__":
    main()