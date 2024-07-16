import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from datasets import DatasetDict

import os

dataset_folder_path = "/home/dell/Desktop/DATASETS/agegender"
train_path = os.path.join(dataset_folder_path,"train")
test_path = os.path.join(dataset_folder_path,"val")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resmi istediğiniz boyuta yeniden boyutlandırın
    transforms.ToTensor(),           # Resmi tensöre dönüştür
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Resmi normalize et
])

train_dataset = ImageFolder(root=train_path, transform=transform)
test_dataset = ImageFolder(root=test_path, transform=transform)

# Dataloaders 
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

dataset = DatasetDict({
    "train": train_dataloader.dataset,
    "test": test_dataloader.dataset
})

print(dataset)


