import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from sklearn.metrics import confusion_matrix
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.classes.LensDataset import LensDataset
from utils.format import *



def training(stats: list, ml_vars: list):
    
    message = f"{DIM}%(asctime)s{RESET}\t\t{BOLD_YELLOW}%(message)s{RESET}"
    dashes = "-" * TERMINALSIZE
    logging.basicConfig(
        level = logging.INFO,
        format = f"\n\n{message}\n{dashes}"
    )
    
    batch_size, learning_rate, num_epochs = ml_vars
    mean, std = stats
    
    num_classes = 2 # Lens vs. non-lens
    num_workers = 4 # GPU/CPU

    # Device will determine whether to run the training on GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use transforms.Compose method to reformat images for modeling, and save to
    # variable all_transforms for later use
    all_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = mean.tolist(),
            std = std.tolist()
        )
    ])

    # Create training, validating, and testing datasets
    train_dataset = LensDataset(
        root_dir = "data/processed/train",
        transform = all_transforms
    )
    validate_dataset = LensDataset(
        root_dir = "data/processed/validate",
        transform = all_transforms
    )
    test_dataset = LensDataset(
        root_dir = "data/processed/test",
        transform = all_transforms
    )
    

    # Create training, validating, and testing loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )

    # Define model
    model = models.resnet18(pretrained=True)
    model.fc = Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss and optimizer
    # Let's use cross-entropy
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n\n\n{BOLD_BLUE}EPOCH {epoch+1}")
        print(f"{"-" * TERMINALSIZE}{RESET}")
        
        # Put the model in training mode
        
        logging.info("Initial training...")
        model.train()
        
        # Running total of the training loss over all samples in this epoch
        # We can use this to compute the average later
        running_loss = 0
        
        # Loop over the training set in batchs
        # The variable images has shape [batch_size, 3, H, W]
        # The variable labels is [batch_size]
        message = f"{"Traning sets":20.20}"
        for images, labels in tqdm(train_loader, desc=message):
            # Move both the image tensors and label tensors to device defined
            # earlier
            images, labels = images.to(device), labels.to(device)
            # Forward pass: run the batch through the CNN
            # The variable predictions has shape [batch_size, 2]
            preds = model(images)
            # Compute cross-entropy loss between predictions and true labels
            loss = criterion(preds, labels)
            
            # Reset gradients that were left over from the previous step
            optimizer.zero_grad()
            # Back-propagate the loss
            loss.backward()
            # Take one optimization step (like an Adam update) using gradients to 
            # adjust the weights
            optimizer.step()
            # This is the total loss across the whole epoch
            running_loss += loss.item() * images.size(0)
        
        # Compute average training loss
        epoch_loss = running_loss / len(train_loader.dataset)
            
        logging.info("Evaluating model...")
        # Validate
        model.eval()
        
        # Used to accumulate validation loss and number of correct predictions
        validate_loss, validate_accuracy = 0.0, 0
        
        # Disable gradient computation to save memory
        with torch.no_grad():
            message = f"{"Validating sets":20.20}"
            for images, labels in tqdm(validate_loader, desc=message):
                # Move validation batch to same device as before
                images, labels = images.to(device), labels.to(device)
                # Forward-pass
                preds = model(images)
                # Compute and accumulate loss like we did in training
                validate_loss += \
                    criterion(preds, labels).item() * images.size(0)
                # Pick the class with highest possibility for each example
                # Compare to labels to get a bool tensor, sum to count correct
                validate_accuracy += (preds.argmax(1) == labels).sum().item()
        
        
        # Average again
        validate_loss /= len(validate_loader.dataset)
        validate_accuracy /= len(validate_loader.dataset)
        
        if validate_accuracy >= 0.95:
            print(f"{GREEN}")
            print(f"Training loss:      {epoch_loss*100:.2f}%")
            print(f"Validating loss:    {validate_loss*100:.2f}%")
            print(f"Accuracy:           {validate_accuracy*100:.2f}%")
            print(f"{RESET}")
            
        elif validate_accuracy >= 0.90:
            print(f"{BLUE}")
            print(f"Training loss:      {epoch_loss*100:.2f}%")
            print(f"Validating loss:    {validate_loss*100:.2f}%")
            print(f"Accuracy:           {validate_accuracy*100:.2f}%")
            print(f"{RESET}")
            
        elif validate_accuracy >= 0.80:
            print(f"{YELLOW}")
            print(f"Training loss:      {epoch_loss*100:.2f}%")
            print(f"Validating loss:    {validate_loss*100:.2f}%")
            print(f"Accuracy:           {validate_accuracy*100:.2f}%")
            print(f"{RESET}")

        else:
            print(f"{RED}")
            print(f"Training loss:      {epoch_loss*100:.2f}%")
            print(f"Validating loss:    {validate_loss*100:.2f}%")
            print(f"Accuracy:           {validate_accuracy*100:.2f}%")
            print(f"{RESET}")
            

    # Test evaluation
    tn_images = []
    tp_images = []
    fn_images = []
    fp_images = []
    
    
    logging.info("Testing model...")
    model.eval()
    
    all_labels = []
    all_predictions = []
    test_accuracy = 0.0
    with torch.no_grad():
        
        message = f"{"Testing sets":20.20}"
        for images, labels in tqdm(test_loader, desc=message):
            
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            test_accuracy += \
                (preds == labels).sum().item() / len(preds)
            all_predictions.extend(preds.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
                
            
            tn = ((labels == 0) & (preds == 0)).nonzero(as_tuple=True)[0]
            for i in tn:
                tn_images.append(images[i])
                
            tp = ((labels == 1) & (preds == 1)).nonzero(as_tuple=True)[0]
            for i in tp:
                tp_images.append(images[i])
            
            fn = ((labels == 1) & (preds == 0)).nonzero(as_tuple=True)[0]
            for i in fn:
                fn_images.append(images[i])
                
            fp = ((labels == 0) & (preds == 1)).nonzero(as_tuple=True)[0]
            for i in fp:
                fp_images.append(images[i])

    cm = confusion_matrix(all_labels, all_predictions, labels=[0,1])


    if test_accuracy >= 0.95:
        print(f"{GREEN}")
        print(f"Test accuracy:      {test_accuracy*100:.2f}%")
        print(f"{RESET}")
        
    elif test_accuracy >= 0.90:
        print(f"{BLUE}")
        print(f"Test accuracy:      {test_accuracy*100:.2f}%")
        print(f"{RESET}")
        
    elif test_accuracy >= 0.80:
        print(f"{YELLOW}")
        print(f"Test accuracy:      {test_accuracy*100:.2f}%")
        print(f"{RESET}")

    else:
        print(f"{RED}")
        print(f"Test accuracy:      {test_accuracy*100:.2f}%")
        print(f"{RESET}")
        
        
    logging.info(f"{BOLD_GREEN}Training complete!{RESET}")
    
    return cm, tn_images, tp_images, fn_images, fp_images