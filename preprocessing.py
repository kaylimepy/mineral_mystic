from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

def load_dataset(root_folder: str) -> ImageFolder:
    '''Loads the dataset from the given root folder and applies data augmentations.'''
    data_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root_folder, transform=data_transformations)

    print('Dataset loaded successfully!')
    return dataset

def split_data(dataset: ImageFolder, train_ratio: float = 0.90, val_ratio: float = 0.05) -> tuple:
    '''Splits the dataset into training, validation, and testing sets based on the provided ratios.'''
    # Create a mapping from index to label
    index_to_label_map = {i: label for i, (_, label) in enumerate(dataset)}
    
    # Split data for validation
    train_indices, val_indices, _, _ = train_test_split(
        list(index_to_label_map.keys()), list(index_to_label_map.values()), 
        stratify=list(index_to_label_map.values()), test_size=val_ratio
    )
    
    # Create a new mapping that excludes validation indices
    index_to_label_map_excluding_val = {idx: label for idx, label in index_to_label_map.items() if idx not in val_indices}
    
    # Calculate the test ratio
    test_ratio = 1 - train_ratio - val_ratio
    
    # Split the remaining data into training and test sets
    train_indices, test_indices, _, _ = train_test_split(
        list(index_to_label_map_excluding_val.keys()), list(index_to_label_map_excluding_val.values()),
        stratify=list(index_to_label_map_excluding_val.values()), test_size=test_ratio
    )
    
    print('Data splitting complete!')
    return train_indices, val_indices, test_indices

def create_dataloaders(dataset: ImageFolder, train_indices: list, val_indices: list, test_indices: list, batch_size: int = 128) -> tuple:
    '''Creates DataLoader objects for training, validation, and testing.'''
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler   = SubsetRandomSampler(val_indices)
    test_sampler  = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader   = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader  = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    print('DataLoaders created successfully!')
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    root_folder = 'temp/mineral_classes'
    dataset     = load_dataset(root_folder)

    train_indices, val_indices, test_indices = split_data(dataset)
    train_loader, val_loader, test_loader    = create_dataloaders(dataset, train_indices, val_indices, test_indices)
    print('Preprocessing complete!')
