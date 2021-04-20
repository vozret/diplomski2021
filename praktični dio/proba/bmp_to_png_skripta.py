from PIL import Image, ImageOps
import glob
import os

from keras.preprocessing.image import load_img

path_to_gt = "/Users/veronika/Documents/FESB_MLID/masks/"
out_dir = "/Users/veronika/Documents/FESB_MLID/png_masks"

gt_dir_paths = sorted(
    [
        os.path.join(path_to_gt, fname)
        for fname in os.listdir(path_to_gt)
        if fname.endswith(".bmp") and not fname.startswith(".")
    ]
)

print("Images:")
for gt_path in gt_dir_paths:
    print(gt_path)

print(len(gt_dir_paths))

for img in glob.glob(path_to_gt + '*.bmp'):
    Image.open(img).save(os.path.join(out_dir, img[len(path_to_gt):len(img)-4] + '.png'))

png_dir_paths = sorted(
    [
        os.path.join(out_dir, fname)
        for fname in os.listdir(out_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("BMP and their PNG:")
for gt_path, png_path in zip(gt_dir_paths, png_dir_paths):
    print(gt_path, "|", png_path)

    