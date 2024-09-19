import torch
from torch import nn
import pytorch_lightning as L
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from pytorch_lightning.loggers import WandbLogger

class CLIPClassifier(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", do_rescale=False)
        self.encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze the encoder and processor
        for param in self.encoder.parameters():
            param.requires_grad = False

        # A lightweight neural network to turn the embeddings into logits in our 101 classes
        self.classifier = nn.Sequential(
                # CLIPVisionModelWithProjection has a default projection_dim of 512
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 101),
                )

        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        with torch.no_grad():
            inputs = self.processor(images = x, return_tensors = "pt")
            inputs.to(device=torch.device("cuda"))
            embeds = self.encoder(**inputs).image_embeds
        results = self.classifier(embeds)
        return results

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = self.loss(predictions, y)
        self.log("test_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = self.loss(predictions, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)






