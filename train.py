import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split

from models import imagebind_model
from models import lora as LoRA
from models.imagebind_model import ModalityType, load_module, save_module
import torch.optim as optim
# Placeholder imports - replace with actual imports from your project
# from your_module import LoRA, imagebind_model, load_module

class AlzheimerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def load_dataset(root_dir):
    image_paths, labels = [], []
    label_dirs = {'ALZHEIMERS_DISEASE': 0, 'COGNITIVE_NORMAL': 1}
    for label_name, label in label_dirs.items():
        label_dir = os.path.join(root_dir, label_name)
        for img_file in os.listdir(label_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(label_dir, img_file))
                labels.append(label)
    return image_paths, labels

class ImageBindTrain(pl.LightningModule):
    def __init__(self, lora_rank, lora_checkpoint_dir, num_classes=2, lr=1e-4, weight_decay=1e-4, lora=False,lora_layer_idxs=None,lora_modality_names=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # Load full pretrained ImageBind model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        if lora:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
            self.model.modality_trunks.update(LoRA.apply_lora_modality_trunks(self.model.modality_trunks, rank=lora_rank,
                                                                              layer_idxs=lora_layer_idxs,
                                                                              modality_names=lora_modality_names))
            LoRA.load_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=lora_checkpoint_dir)

            # Load postprocessors & heads
            load_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=lora_checkpoint_dir)
            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
        elif linear_probing:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
            for modality_postprocessor in self.model.modality_postprocessors.children():
                modality_postprocessor.requires_grad_(False)

            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
            for modality_head in self.model.modality_heads.children():
                modality_head.requires_grad_(False)
                final_layer = list(modality_head.children())[-1]
                final_layer.requires_grad_(True)
				

        self.classifier = nn.Linear(1024, num_classes)
        self.accuracy = Accuracy(num_classes=num_classes,average='macro',task='multiclass')
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        inputs={'vision':x}
        features=self.model(inputs)
        output_features=features.get('vision')
        if output_features is not None:
            return self.classifier(output_features)
        else:
            raise ValueError("Model did not return features for the specified modality")

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

# Include the argparse, train_model function and if __name__ == "__main__": block here as before.
def parse_args():
    parser = argparse.ArgumentParser(description="Train the ImageBind model with PyTorch Lightning and LoRA.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training ('cpu' or 'cuda')")
    parser.add_argument("--datasets_dir", type=str, default="./Dataset",
                        help="Directory containing the datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["patientsdump"], choices=["patientsdump"],
                        help="Datasets to use for training and validation")
    parser.add_argument("--full_model_checkpoint_dir", type=str, default=".checkpoints/full",
                        help="Directory to save the full model checkpoints")
    parser.add_argument("--full_model_checkpointing", action="store_true", help="Save full model checkpoints")
    parser.add_argument("--loggers", type=str, nargs="+", choices=["tensorboard", "wandb", "comet", "mlflow"],
                        help="Loggers to use for logging")
    parser.add_argument("--loggers_dir", type=str, default="./.logs", help="Directory to save the logs")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (Don't plot samples on start)")

    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--self_contrast", action="store_true", help="Use self-contrast on the image modality")

    parser.add_argument("--lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA layers")
    parser.add_argument("--lora_checkpoint_dir", type=str, default="./.checkpoints/lora",
                        help="Directory to save LoRA checkpoint")
    parser.add_argument("--lora_modality_names", nargs="+", type=str, default=["vision", "text"],
                        choices=["vision", "text", "audio", "thermal", "depth", "imu"],
                        help="Modality names to apply LoRA")
    parser.add_argument("--lora_layer_idxs", nargs="+", type=int,
                        help="Layer indices to apply LoRA")
    parser.add_argument("--lora_layer_idxs_vision", nargs="+", type=int,
                        help="Layer indices to apply LoRA for vision modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_text", nargs="+", type=int,
                        help="Layer indices to apply LoRA for text modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_audio", nargs="+", type=int,
                        help="Layer indices to apply LoRA for audio modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_thermal", nargs="+", type=int,
                        help="Layer indices to apply LoRA for thermal modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_depth", nargs="+", type=int,
                        help="Layer indices to apply LoRA for depth modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_imu", nargs="+", type=int,
                        help="Layer indices to apply LoRA for imu modality. Overrides lora_layer_idxs if specified")

    parser.add_argument("--linear_probing", action="store_true",
                        help="Freeze model and train the last layers of the head for each modality.")

    return parser.parse_args()

def train_model(args):
    # Setup based on passed arguments
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths, labels = load_dataset(args.datasets_dir)
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2)

    train_dataset = AlzheimerDataset(train_image_paths, train_labels, transform=transform)
    val_dataset = AlzheimerDataset(val_image_paths, val_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Parse indices of layers to apply LoRA
    lora_layer_idxs = {}
    lora_modality_names = []
    modalities = ["vision", "text", "audio", "thermal", "depth", "imu"]
    for modality_name in args.lora_modality_names:
        if modality_name in modalities:
            modality_type = getattr(ModalityType, modality_name.upper())
            lora_layer_idxs[modality_type] = getattr(args, f'lora_layer_idxs_{modality_name}', None)
            if not lora_layer_idxs[modality_type]:
                lora_layer_idxs[modality_type] = None
            lora_modality_names.append(modality_type)
        else:
            raise ValueError(f"Unknown modality name: {modality_name}")

    model = ImageBindTrain(lora_rank=args.lora_rank, lora_checkpoint_dir=args.lora_checkpoint_dir, lora=args.lora, num_classes=2, lr=args.lr, weight_decay=args.weight_decay)

    # Set up logger
    loggers = []
    if 'tensorboard' in args.loggers:
        loggers.append(TensorBoardLogger("tb_logs", name="my_model"))
    if 'comet' in args.loggers:
        loggers.append(CometLogger(
            api_key="TL29lVHGhSIwuQXeKLmBgZgak",
            workspace="sripad-tech",  # Optional
            project_name="general",
        ))
		
	# Define ModelCheckpoint with custom directory and filename format
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='.checkpoints/full',  # Specify your custom path here
        filename='{epoch}-{val_loss:.8f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() and 'cuda' in args.device else "cpu",
        devices=1 if torch.cuda.is_available() and  "cuda" in args.device else None,
        logger=loggers if loggers else None,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
