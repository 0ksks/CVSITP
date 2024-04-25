import os
from tqdm import tqdm

def get_mask_label_name(imgRootPath)->dict[str, list[map]]:
    files = os.listdir(imgRootPath)
    ret = {}
    for file in files:
        dirPath = os.path.join(imgRootPath, file)
        if os.path.isdir(dirPath):
            imgs = os.listdir(dirPath)
            masks = map(lambda x:x.replace("jpg","png"), filter(lambda x:"jpg" in x, imgs))
            labels = map(lambda x:x.replace("jpg","txt"), filter(lambda x:"jpg" in x, imgs))
            ret[file] = [masks, labels]
    return ret

def mask_to_detect_label(maskPath:str, labelPath:str):
    import numpy as np
    from PIL import Image
    
    mask = Image.open(maskPath)
    mask = np.array(mask)[:,:,0]

    H, W = mask.shape

    def mask_to_yolo(mask, class_index):
        points = np.argwhere(mask == class_index)
        if len(points) == 0:
            return None

        x_min = min(points[:, 1]) / W
        y_min = min(points[:, 0]) / H
        x_max = max(points[:, 1]) / W
        y_max = max(points[:, 0]) / H

        return [class_index-1, x_min, y_min, x_max, y_max]

    with open(labelPath, "w") as f:
        for class_index in range(1, np.max(mask) + 1):
            yolo_label = mask_to_yolo(mask, class_index)
            if yolo_label is not None:
                f.write(" ".join(map(str, yolo_label)) + "\n")

def transform(imgRootPath:str, labelRootPath:str, originPath:str):
    mask_label_name = get_mask_label_name(imgRootPath)
    for k in mask_label_name.keys():
        k = "val"
        masks, labels = mask_label_name[k]
        for mask, label in tqdm(zip(masks, labels), desc=f"{k}"):
            maskPath = os.path.join(originPath, k)
            maskPath = os.path.join(maskPath, "mask")
            maskPath = os.path.join(maskPath, mask)
            labelPath = os.path.join(labelRootPath, k)
            os.makedirs(labelPath, exist_ok=True)
            labelPath = os.path.join(labelPath, label)
            mask_to_detect_label(maskPath, labelPath)
        return

if __name__ == "__main__":
    ...
    transform("process/detect_data/images", "process/detect_data/labels", "UECFOODPIXCOMPLETE/UECFoodPIXCOMPLETE")