#coding:utf-8
import sys
import PIL.Image as Image
import numpy as np

labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# rrrgggbbb
def see_raw(file, h, w):
    raw = np.fromfile(file, dtype=np.uint8)
    print(raw[0], labels[raw[0]])
    raw = raw[1:]
    raw.shape = (3, h, w)
    raw = np.transpose(raw, (1,2,0))

    img = Image.fromarray(raw)

    img.show()

def main():
    file = sys.argv[1]
    h = 32
    w = 32
    print((h,w))
    see_raw(file, h, w)


if __name__ == "__main__":
    main()