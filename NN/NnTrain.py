from copy import deepcopy
def kernel(modelHyperParams=None,
           modelStaticParams=None,
           loaderHyperParams=None,
           loaderStaticParams=None,
           miniLoaderParams=None,
           trainerParams=None,):
    from DataLoader import split_dataloader, mini_dataloader
    from NnStructure import get_model
    from pytorch_lightning import Trainer
    from lightning.pytorch.loggers import TensorBoardLogger
    def stringfyParams(params:dict):
        paramsLocal = deepcopy(params)
        if paramsLocal=={}:
            return "default"
        for k,v in paramsLocal.items():
            if isinstance(v, float):
                if v>1e3 or v<1e-3:
                    paramsLocal[k] = f"{v:.3e}"
                else:
                    paramsLocal[k] = round(v, 3)
            if isinstance(v, dict):
                paramsLocal[k] = "".join([f"{kk}[{vv}]".replace("/",".") for kk,vv in v.items() if vv is not None])
        return "@".join([f"{k}[{v}]" for k,v in paramsLocal.items()])
    loaders = split_dataloader(**loaderHyperParams, **loaderStaticParams)
    trainLoader = loaders[0]
    if miniLoaderParams!={}:
        trainLoader = mini_dataloader(dataloader=trainLoader, **miniLoaderParams)
    print("----------MODEL INFO----------")
    print(f"model  H | {stringfyParams(modelHyperParams)}")
    print(f"loader H | {stringfyParams(loaderHyperParams)}")
    print(f"model  S | {stringfyParams(modelStaticParams)}")
    print(f"loader S | {stringfyParams(loaderStaticParams)}")
    print(f"trainer  | {stringfyParams(trainerParams)}")
    print(f"datasize | {len(trainLoader.dataset)}")
    model = get_model(**modelHyperParams, **modelStaticParams)
    if trainerParams["logger"]:
        trainerParams["logger"] = TensorBoardLogger(save_dir=f"hyperLogs/{stringfyParams(modelHyperParams)}/")
    trainer = Trainer(**trainerParams)
    trainer.fit(model, trainLoader)
    train_loss = trainer.callback_metrics["train_loss"].item()
    return train_loss
if __name__ == "__main__":
    params = {
        "modelHyperParams":{
            "coreNetName":"EfficientNet",
        },
        "modelStaticParams":{
            "optim_":"Adam",
            "classes":256
        },
        "loaderHyperParams":{
            "batchSize":10
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
            "enable_checkpointing":False
        }
    }
    kernel(**params)