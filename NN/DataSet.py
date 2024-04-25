# print("import torchvision.datasets")
from torchvision import datasets
from CVDataEnhance import get_transform

def get_dataset(params):
    trainTrans, testTrans = get_transform(params["centerCrop"], params["resize"])
    trainImageDataset = datasets.ImageFolder(params["root"], trainTrans)
    testImageDataset = datasets.ImageFolder(params["root"], testTrans)
    return trainImageDataset, testImageDataset

if __name__ == "__main__":
    params = {
        "root":"data/UECFOOD256",
        "centerCrop":20,
        "resize":None
        }
    trainSet, testSet = get_dataset(params)
    print(f"{len(trainSet) = }")
    print(trainSet[0][0].shape, trainSet[0][1])