import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import numpy as np
from skimage import color

class ColorPredictor(nn.Module):
    def __init__(self, num_outputs=15):
        super(ColorPredictor, self).__init__()
        self.base = models.resnet18()
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

def denormalize_lab(norm_lab):
    l_norm, a_norm, b_norm = norm_lab
    return [l_norm * 100.0, a_norm * 255.0 - 128, b_norm * 255.0 - 128]

def lab_to_rgb(lab):
    lab_array = np.array(lab, dtype=np.float64).reshape(1, 1, 3)
    rgb_float = color.lab2rgb(lab_array)
    return tuple((np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)[0][0])

def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Predyktor Dominujących Kolorów")
        self.root.geometry("600x400")

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        top_frame = tk.Frame(root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)

        model_frame = tk.LabelFrame(top_frame, text="1. Wybór modelu")
        model_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_models = tk.Button(model_frame, text="Wybierz folder generacji", command=self.select_model_folder)
        btn_models.pack(pady=5, padx=5, fill=tk.X)
        self.model_combobox = ttk.Combobox(model_frame, state="disabled")
        self.model_combobox.pack(pady=5, padx=5, fill=tk.X)
        
        image_frame = tk.LabelFrame(top_frame, text="2. Wybór obrazu")
        image_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_image = tk.Button(image_frame, text="Wybierz obraz", command=self.select_image)
        btn_image.pack(pady=5, padx=5, fill=tk.X)
        self.lbl_image_path = tk.Label(image_frame, text="Nie wybrano obrazu", fg="gray", width=25, anchor='w')
        self.lbl_image_path.pack(pady=5, padx=5)

        btn_predict = tk.Button(root, text="3. Wskaż dominujące kolory", command=self.run_prediction, font=('Helvetica', 12, 'bold'), bg='lightblue')
        btn_predict.pack(fill=tk.X, padx=15, pady=5)
        
        results_container = tk.Frame(root)
        results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_canvas = tk.Canvas(results_container, bg='lightgray', width=224, height=224)
        self.image_canvas.pack(side=tk.LEFT, padx=10, anchor='n')

        self.colors_frame = tk.Frame(results_container)
        self.colors_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.color_canvases = []
        for i in range(5):
            canvas = tk.Canvas(self.colors_frame, bg='white', height=40, highlightthickness=1, highlightbackground="black")
            canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0 if i > 0 else 5))
            self.color_canvases.append(canvas)

    def select_model_folder(self):
        path = filedialog.askdirectory(initialdir="./models", title="Wybierz folder z modelami")
        if path:
            self.model_folder_path = path
            models_list = [f for f in os.listdir(path) if f.endswith(".pth")]
            if models_list:
                self.model_combobox['values'] = models_list
                self.model_combobox.set(models_list[0])
                self.model_combobox.config(state="readonly")
            else:
                messagebox.showwarning("Brak modeli", "W wybranym folderze nie znaleziono plików '.pth'")

    def select_image(self):
        path = filedialog.askopenfilename(title="Wybierz obraz", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            self.image_path = path
            self.lbl_image_path.config(text=os.path.basename(path))
            img = Image.open(path)
            img.thumbnail((224, 224))
            self.photo_img = ImageTk.PhotoImage(img)
            self.image_canvas.create_image(112, 112, image=self.photo_img)

    def run_prediction(self):
        if not getattr(self, 'model_folder_path', None) or not self.model_combobox.get():
            messagebox.showerror("Błąd", "Proszę wybrać folder i plik modelu.")
            return
        if not getattr(self, 'image_path', None):
            messagebox.showerror("Błąd", "Proszę wybrać obraz.")
            return

        try:
            model_path = os.path.join(self.model_folder_path, self.model_combobox.get())
            self.model = ColorPredictor(num_outputs=15).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(image_tensor)
            
            predicted_norm_lab_flat = output.cpu().numpy()[0]
            predicted_norm_lab_reshaped = predicted_norm_lab_flat.reshape(5, 3)

            for i, norm_lab in enumerate(predicted_norm_lab_reshaped):
                lab = denormalize_lab(norm_lab)
                rgb = lab_to_rgb(lab)
                hex_val = rgb_to_hex(rgb)
                self.color_canvases[i].config(bg=hex_val)

        except Exception as e:
            messagebox.showerror("Błąd predykcji", f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()