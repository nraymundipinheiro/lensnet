from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset



class LensDataset(Dataset):
    """
    PyTorch Dataset for strong-lens vs. non-lens images.
    """
    
    def __init__(self, root_dir, transform=None):
        # Directory with all the images
        self.root_dir = Path(root_dir)
        # Optional transform to be applied on a sample
        self.transform = transform
        # Biold (path, label) pairs
        self.samples = []
        
        class_options = [
            ("positive", 1),
            ("negative", 0)
        ]
        
        for label_name, label_num in class_options:
            class_dir = self.root_dir / label_name
            
            for image_path in class_dir.glob("*.png"):
                self.samples.append((
                    image_path,
                    label_num
                ))
            
    
    def __len__(self):
        return len(self.samples)
    
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        
        return image, label
        