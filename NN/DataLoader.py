# print("import dataset, dataloader")
from DataSet import get_dataset as __get_dataset
from torch.utils.data import random_split, DataLoader, Subset
import torch
from typing import Union


BATCH_SIZE = 10
SHUFFLE = True
RANDOM_SEED = 2003
CENTER_CROP = 224
RESIZE = None
CLASSES = 256

DATASET_PARAMS = {
    "root":"data/UECFOOD256",
    "centerCrop":CENTER_CROP,
    "resize":RESIZE
}

def split_dataloader(
    ratio:tuple=(8, 1, 1),
    batchSize:Union[int, list[int]]=BATCH_SIZE,
    shuffle=SHUFFLE,
    datasetParams=DATASET_PARAMS,
    randomSeed = RANDOM_SEED,
    ) -> list[DataLoader]:
    '''
    [train, val, test]
    '''
    generator=torch.Generator().manual_seed(randomSeed)
    # print(f"split by {ratio}")
    if isinstance(batchSize, int):
        batchSize = [batchSize,]*3
    ratio = list(map(lambda x:x/sum(ratio), ratio))
    ratio[-1] = 1 - sum(ratio[:-1])
    datasets = __get_dataset(datasetParams)
    trainDatasets = random_split(datasets[0], ratio, generator)
    testDatasets = random_split(datasets[1], ratio, generator)
    dataloaders = []
    for idx in range(len(ratio)):
        if idx==0:
            dataloaders.append(DataLoader(trainDatasets[idx], batchSize[idx], shuffle))
        else:
            dataloaders.append(DataLoader(testDatasets[idx], batchSize[idx]))
    return dataloaders

def mini_dataloader(dataloader:DataLoader, subSize=100, subBatchSize=10, subShuffle=True):
    oriDataset = dataloader.dataset
    oriDataSize = len(oriDataset)
    import random
    randomIndices = random.sample(range(oriDataSize), subSize)
    return DataLoader(Subset(oriDataset, randomIndices), subBatchSize, shuffle=subShuffle)

if __name__ == "__main__":
    a:list[DataLoader] = split_dataloader()
    img, label = a[1].dataset[0]
    print(img.shape, label)
    