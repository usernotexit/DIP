# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_facades import FacadesDataset
from dataset_edge2shoes import DatasetEdge2Shoes
from network_GAN import Generator, Discriminator
from torch.optim.lr_scheduler import StepLR

import argparse

parser = argparse.ArgumentParser(description='Pix2pix Segmention Training')
parser.add_argument('--datasets_folder', required=True, help='数据文件夹')
parser.add_argument('--output_folder', default='/output', help='folder to output images and model checkpoints')
parser.add_argument('--batch_size', type=int, default=10, help='batch size, set to 1 on PC')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--L1_lambda', type=float, default=100, help='L1正则项系数')
parser.add_argument('--nData', type=int, default=1000, help='仅选取部分数据训练')
parser.add_argument('--type', default='facades', help='数据集类型，默认facades')
args = parser.parse_args()

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    num_images = min(num_images, 10)
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(G, D, lamb, dataloader, optimizer_G, optimizer_D, criterion_G, criterion_D, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        G, D (nn.Module): The neural network model: generator and discriminator.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer_G, _D (Optimizer): Optimizer for updating model parameters.
        criterion_G, _D (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    G.train()
    D.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        optimizer_D.zero_grad()

        # Train D
        # use G to make fake imgs, train D with both fake and real imgs
        #dde = torch.nn.LeaklyReLU(0.2)
        #dde(image_rgb)
        output_D = D(image_rgb, image_semantic).squeeze()
        loss_real_D = criterion_D(output_D, torch.ones(output_D.size()).to(device))
        #G.eval()
        output_G = G(image_semantic)
        output_D = D(output_G, image_semantic).squeeze()
        loss_fake_D = criterion_D(output_D, torch.zeros(output_D.size()).to(device))

        loss_D = (loss_real_D + loss_fake_D) * 0.5
        loss_D.backward()
        optimizer_D.step()

        running_loss_D += loss_D.item()

        # Save sample images every 5 epochs
        if epoch % 10 == 0 and i == 0:
            save_images(image_semantic, image_rgb, output_G, f'{args.output_folder}/train_results', epoch, args.batch_size)

        # Train G
        # loss_G contains of L1loss with output_G & real imgs and loss of D(output_G) with 'true'
        #G.train()
        G.zero_grad()
        output_G = G(image_semantic)
        output_D = D(output_G, image_semantic)
        loss_G = criterion_D(output_D, torch.ones(output_D.size()).to(device)) + lamb * criterion_G(output_G, image_rgb)

        # Backward pass and optimization
        loss_G.backward()
        optimizer_G.step()

        # Update running loss
        running_loss_G += loss_G.item()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Generator Loss: {loss_G.item():.4f}, Discriminator Loss: {loss_D.item():.4f}')

def validate(G, D, lamb, dataloader, criterion_G, criterion_D, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    G.eval()
    D.eval()
    val_loss_G = 0.0
    val_loss_D = 0.0
    correct_D_real = 0
    correct_D_fake = 0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            output_G = G(image_semantic)
            output_D_real = D(image_rgb, image_semantic)
            output_D_fake = D(output_G, image_semantic)

            # Compute the loss
            loss_G = lamb * criterion_G(output_G, image_semantic) + criterion_D(output_D_fake, torch.ones(output_D_fake.size()).to(device))
            loss_D = (criterion_D(output_D_real, torch.ones(output_D_real.size()).to(device)) + criterion_D(output_D_fake, torch.zeros(output_D_fake.size()).to(device))) * 0.5

            val_loss_G += loss_G.item()
            val_loss_D += loss_D.item()

            # Save sample images every 5 epochs
            if epoch % 10 == 0 and i == 0:
                save_images(image_semantic, image_rgb, output_G, f'{args.output_folder}/val_results', epoch, args.batch_size)

            # Correctness of the Discriminator
            predictions_real = torch.round(output_D_real)  # Assuming the output is in the range [0, 1]
            predictions_fake = torch.round(output_D_fake)

            correct_D_real += (predictions_real == 1).sum().item() / predictions_real.numel()
            correct_D_fake += (predictions_fake == 0).sum().item() / predictions_fake.numel()
            
    # Calculate average validation loss
    avg_val_loss_G = val_loss_G / len(dataloader)
    avg_val_loss_D = val_loss_D / len(dataloader)
    accuracy_D_real = correct_D_real / len(dataloader)
    accuracy_D_fake = correct_D_fake / len(dataloader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss G: {avg_val_loss_G:.4f}, Loss D: {avg_val_loss_D:.4f}, Acc D (Real): {accuracy_D_real:.4f}, Acc D (Fake): {accuracy_D_fake:.4f}')            
if __name__ == '__main__':
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    if args.type == 'edge2shoes':
        train_dataset = DatasetEdge2Shoes(args.datasets_folder+'/train', args.nData)
        val_dataset = DatasetEdge2Shoes(args.datasets_folder+'/val', 200)
    else:
        train_dataset = FacadesDataset(args.datasets_folder, list_file='train_list.txt')
        val_dataset = FacadesDataset(args.datasets_folder, list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    G = Generator().to(device)
    D = Discriminator().to(device)
    criterion_G = nn.L1Loss()
    criterion_D = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.2)
    scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.2)

    # Training loop
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        train_one_epoch(G, D, args.L1_lambda, train_loader, optimizer_G, optimizer_D, criterion_G, criterion_D, device, epoch, num_epochs)
        validate(G, D, args.L1_lambda, val_loader, criterion_G, criterion_D, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler_G.step()
        scheduler_D.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs(f'{args.output_folder}/checkpoints', exist_ok=True)
            torch.save(G.state_dict(), f'{args.output_folder}/checkpoints/G_model_epoch_{epoch + 1}.pth')
            torch.save(D.state_dict(), f'{args.output_folder}/checkpoints/D_model_epoch_{epoch + 1}.pth')

    os.makedirs(f'{args.output_folder}/checkpoints', exist_ok=True)
    torch.save(G.state_dict(), f'{args.output_folder}/checkpoints/G_model_final.pth')
    torch.save(D.state_dict(), f'{args.output_folder}/checkpoints/D_model_final.pth')
