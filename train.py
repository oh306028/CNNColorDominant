import os
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
from skimage import color

#Configuration 
IMAGE_DIR = "./images"
LABEL_FILE = "./labels.txt"
BASE_MODELS_DIR = "models"
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2

#Colors modifiers
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb):
    rgb_float = np.array(rgb, dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb_float.reshape(1, 1, 3))[0][0]
    return lab

def normalize_lab(lab):
    l, a, b = lab
    return [l / 100.0, (a + 128) / 255.0, (b + 128) / 255.0]

def get_new_attempt_folder(base_dir=BASE_MODELS_DIR, prefix="attempt"):
    os.makedirs(base_dir, exist_ok=True)
    i = 0
    while True:
        attempt_path = os.path.join(base_dir, f"{prefix}{i}")
        if not os.path.exists(attempt_path):
            os.makedirs(attempt_path)
            return attempt_path
        i += 1

#Dataset
class ColorDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                
                filename = parts[0]
                hex_colors_str = " ".join(parts[1:])
                hex_colors = [h.strip() for h in hex_colors_str.split(',')]

                if len(hex_colors) != 5:
                    continue

                lab_colors_flat = []
                for hex_code in hex_colors:
                    rgb = hex_to_rgb(hex_code)
                    lab = rgb_to_lab(rgb)
                    lab_norm = normalize_lab(lab)
                    lab_colors_flat.extend(lab_norm)
                
                self.samples.append((filename, lab_colors_flat))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, lab_colors = self.samples[idx]
        path = os.path.join(self.image_dir, filename)
        try:
            image = Image.open(path).convert('RGB')
        except FileNotFoundError:
            print(f"Błąd: Nie znaleziono pliku: {path}")
            return torch.empty(0), torch.empty(0)
            
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(lab_colors, dtype=torch.float32)
        return image, label

#Model
class ColorPredictor(nn.Module):
    def __init__(self, num_outputs=15): # 5 kolorów * 3 wartości LAB
        super(ColorPredictor, self).__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        for param in self.base.parameters():
            param.requires_grad = False
            
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base(x)

def train():
    models_folder = get_new_attempt_folder()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ColorDataset(IMAGE_DIR, LABEL_FILE, transform)
    if not full_dataset:
        print("Nie wczytano żadnych danych. Sprawdź plik labels.txt i folder images.")
        return

    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Rozmiar zbioru treningowego: {len(train_dataset)}")
    print(f"Rozmiar zbioru walidacyjnego: {len(val_dataset)}")
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x[0].numel() > 0, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else (torch.empty(0), torch.empty(0))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ColorPredictor(num_outputs=15).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            if images.numel() == 0: continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                if images.numel() == 0: continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(models_folder, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Zapisano nowy najlepszy model z Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(LABEL_FILE):
        print("Błąd: Upewnij się, że folder 'images' i plik 'labels.txt' istnieją w głównym katalogu.")
    else:
        train()