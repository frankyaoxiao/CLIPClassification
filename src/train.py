from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import CLIPClassifier
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import os

dir = os.getcwd()
batch_size = 64
img_size = (250, 250)
split = [.7, .2, .1] # Train, Test, Val
num_epochs = 30
lr = 1e-2
workers = 3
transforms = transforms.Compose([
    transforms.Resize(size = img_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x), # Deal with grayscale images
    ])

dataset = datasets.Caltech101(root=dir, download=True, transform=transforms)

train_data, test_data, val_data = random_split(dataset, split)

train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=workers)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=workers)
val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=workers)
model = CLIPClassifier(lr)
trainer = L.Trainer(
        logger=WandbLogger(name="clip"),
        max_epochs=num_epochs,
        )

trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(model, test_dataloader)





