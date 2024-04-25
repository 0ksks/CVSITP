import numpy as np
from collections import defaultdict
from json import dumps,loads
from typing import Literal

TUNABLE = None

def search(lrRange:list, batchSizeRange:list, coreNetName:Literal["EfficientNet","ViT"], params):
    '''
    lrRange:[start, end, steps]
    batchSizeRange:[sizeA, sizeB, ...]
    '''
    params["modelHyperParams"]["coreNetName"] = coreNetName
    from NnTrain import kernel
    result = defaultdict(list)
    for batchSizeIdx, batchSize in enumerate(batchSizeRange):
        for lrIdx, lr in enumerate(np.linspace(*lrRange)):
            lr = np.float_power(10, lr)
            print("----------SEARCH PROGRESS----------")
            print(f"coreNet[{coreNetName}]@batchSize[{batchSizeIdx+1}/{len(batchSizeRange)}]@lr[{lrIdx+1}/{lrRange[2]}]")
            params["loaderHyperParams"]["batchSize"] = batchSize
            params["modelHyperParams"]["lr"] = lr
            result[f"batchSize_{batchSize}"].append(kernel(**params))
    result["lrRange"] = lrRange
    with open(f"NN/{coreNetName}_batchSize_LR.json","w") as f:
        f.write(dumps(result))

def plot_result():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    with open("batchSizeLR.json","r") as f:
        result:dict = loads(f.read())
    lr = np.linspace(*result["lrRange"])
    labels = []
    for key in result.keys():
        if key!="lrRange":
            labels.append(key)
            plt.plot(lr, result[key])
    lrTicks = list(lr)[::len(lr)//10]
    lrTicks.append(lr[-1])
    plt.xticks(lrTicks, map(lambda x:f"{round(x,3)}", lrTicks))
    plt.xlabel("log10(learning rate)")
    plt.ylabel("loss")
    plt.legend(labels,loc="lower left")
    plt.title("Best Learning Rate")
    plt.savefig("BestBatchLr.png")

if __name__ == "__main__":
    lrRange = [-5, -1, 100]
    batchSizeRange = [64, 128, 256]
    coreNetNameList = ["EfficientNet", "ViT"]
    params = {
        "modelHyperParams":{
            "coreNetName":TUNABLE,
            "lr":TUNABLE
        },
        "modelStaticParams":{
            "optim_":"Adam",
            "classes":256
        },
        "loaderHyperParams":{
            "batchSize":TUNABLE
        },
        "loaderStaticParams":{
            "ratio":(8,1,1),
            "shuffle":True,
            "datasetParams":{
                "root":"data/UECFOOD256",
                "centerCrop":224,
                "resize":None
            },
            "randomSeed":2003
        },
        "miniLoaderParams":{
            "subSize":10,
            "subBatchSize":5,
            "subShuffle":True
        },
        "trainerParams":{
            "max_epochs":1,
            "logger":False,
            "accelerator":"gpu",
            "devices":-1,
            "enable_checkpointing":False,
            "enable_progress_bar":False,
            "enable_model_summary":False
        }
    }
    for coreNetName in coreNetNameList:
        search(lrRange, batchSizeRange, coreNetName, params)