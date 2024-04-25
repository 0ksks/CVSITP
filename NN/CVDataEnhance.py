# print("import torchvision.transforms, torch")
import torchvision.transforms as trans
import torch
def get_transform(centerCrop:int, resize:tuple[int,int]=None):
    '''
    [train, test]
    '''
    trainTransformer = [
            trans.RandomRotation(45),
            trans.CenterCrop(centerCrop),
            trans.RandomHorizontalFlip(p=0.5),
            trans.RandomVerticalFlip(p=0.5),
            trans.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    if resize is not None:
        trainTransformer = [trans.Resize(resize),] + trainTransformer
    
    testTransformer = [
            trans.Resize([centerCrop, centerCrop]),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    
    transformConfig = [
        trans.Compose(trainTransformer),
        trans.Compose(testTransformer)
    ]
    return transformConfig
if __name__ == "__main__":
    a = get_transform(20,[40,50])[0]
    img = torch.randint(0, 256, size=(3, 90, 90), dtype=torch.uint8)
    print(img.shape)
    img = a(img)
    print(img.shape)