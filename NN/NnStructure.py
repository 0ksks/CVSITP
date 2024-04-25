# print("import nn, optim, pl")
import torch.optim as optim
import torch
import pytorch_lightning as pl
from typing import Literal

import torch.utils

class Model(pl.LightningModule):
    def __init__(self, coreNet, optim_, lr:float=0.001) -> None:
        super(Model, self).__init__()
        self.save_hyperparameters(ignore=["coreNet"])
        self.coreNet = coreNet
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.test_step_outputs = []
        self.optim = optim_
        # print(f"lr={lr}\nloss_fn={self.loss_fn}")
    def forward(self, input_):
        return self.coreNet(input_)
    def training_step(self, batch, batch_idx):
        input_, target = batch
        output = self.forward(input_)
        loss:torch.Tensor = self.loss_fn(output, target)
        self.log("train_loss", loss)
        # if batch_idx==0:
        #     print(loss.item())
        return loss
    def validation_step(self, batch, batch_idx):
        input_, target = batch
        output = self.forward(input_)
        loss:torch.Tensor = self.loss_fn(output, target)
        self.log("validation_loss", loss)
        return loss
    def test_step(self, batch, batch_idx):
        input_, target = batch
        output = self.forward(input_)
        loss:torch.Tensor = self.loss_fn(output, target)
        self.test_step_outputs.append(loss)
        return loss
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs).mean()
        self.log('avg_test_loss', avg_loss)
        self.test_step_outputs.clear()
    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr)
        return optimizer

def get_model(coreNetName:Literal["EfficientNet","ViT"], optim_:Literal["Adam","SGD"], classes:int, lr:float=1e-3):

    if optim_=="Adam":
        optim_ = optim.Adam
    elif optim_=="SGD":
        optim_ = optim.SGD

    if coreNetName=="EfficientNet":
        from efficientnet_pytorch import EfficientNet
        efficientNet = EfficientNet.from_pretrained("efficientnet-b0", weights_path="NN/efficientnet-b0-355c32eb.pth", num_classes=classes)
        coreNet = efficientNet
        
    elif coreNetName=="ViT":
        from vit_model import vit_base_patch32_224_in21k
        vit = vit_base_patch32_224_in21k(num_classes=classes)
        state_dict = torch.load("NN/jx_vit_base_patch32_224_in21k-8db57226.pth")
        state_dict['head.weight'] = state_dict['head.weight'][:256, :]
        state_dict['head.bias'] = state_dict['head.bias'][:256]
        vit.load_state_dict(state_dict)
        coreNet = vit
    return Model(
        coreNet=coreNet,
        optim_=optim_,
        lr=lr
        )

if __name__=="__main__":
    inputTensor = torch.randn((2,3))
    model = Model()
    outputTensor = model(inputTensor)
    print(f"{inputTensor = }")
    print(f"{outputTensor = }")