import numpy as np
import warnings

from slsim.Plots.plot_functions import create_image_montage_from_image_list

from src.preprocess import preprocess
from src.train import training

warnings.filterwarnings("ignore")



def main():
    split_fractions = [0.70, 0.15, 0.15]
    stats = preprocess(split_fractions)
    
    batch_size, learning_rate, num_epochs = 32, 0.001, 10
    
    tn_images, tp_images, fn_images, fp_images = \
        training(stats, [batch_size, learning_rate, num_epochs])
    
    print(f"TN:\n{tn_images}")
    print(f"TP:\n{tp_images}")
    print(f"FN:\n{fn_images}")
    print(f"FP:\n{fp_images}")
    
    
    
    

    # 1) Helper to convert a normalized tensor to an (H,W,3) array in [0,1]
    def tensor_to_rgb(img_t):
        # img_t: torch.Tensor of shape (3,H,W), values ∈ [0,1]
        arr = img_t.cpu().numpy()                 # -> (3,H,W)
        arr = np.transpose(arr, (1,2,0))          # -> (H,W,3)
        return arr

    for images in tn_images, tp_images, fn_images, fp_images:
        # 2) Build a Python list of your first N mis-classified images
        rgb_list = [ tensor_to_rgb(t) for t in images[:100] ]  # e.g. first 100

        # 3) Create the montage
        montage = create_image_montage_from_image_list(
            num_rows=10,                # e.g. 10 rows × 10 cols = 100 images
            num_cols=10,
            images=rgb_list,
            time=None
        )
        
    
    
if __name__ == "__main__":
    main()