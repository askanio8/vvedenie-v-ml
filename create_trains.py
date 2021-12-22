import random

from PIL import Image
import glob, os

size = 256, 256

for infile in glob.glob("F:/pl/*.jpg"):
    file, ext = os.path.splitext(infile)
    with Image.open(infile) as im:
        im = im.crop((0, 0, min(im.size), min(im.size)))
        im = im.resize(size, Image.ANTIALIAS)
        im.save("F:/th\\"+ str(random.randint(1000,1000000)) + ".jpg", "JPEG")