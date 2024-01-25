import os
import numpy as np
from torchvision import datasets, transforms
from pytorch_fid import fid_score, inception_score

def calculate_fid_score(real_folder, fake_folder):
    real_dataset = datasets.ImageFolder(root=real_folder, transform=transforms.ToTensor())
    fake_dataset = datasets.ImageFolder(root=fake_folder, transform=transforms.ToTensor())

    fid_value = fid_score.calculate_fid_given_paths([real_dataset, fake_dataset], batch_size=50, cuda=False)
    return fid_value

def calculate_is_score(folder):
    dataset = datasets.ImageFolder(root=folder, transform=transforms.ToTensor())
    is_mean, is_std = inception_score.calculate_inception_score(dataset, cuda=False)
    return is_mean

def main():
    standard_folder = 'standard'
    groups = ['group1', 'group2', 'group3', 'group4', 'group5']

    for group in groups:
        fid_score_value = calculate_fid_score(os.path.join(standard_folder, 'images'), os.path.join(group, 'images'))
        is_score_value = calculate_is_score(os.path.join(group, 'images'))

        print(f'Group {group}:')
        print(f'FID Score: {fid_score_value}')
        print(f'Inception Score: {is_score_value}')

if __name__ == "__main__":
    main()
