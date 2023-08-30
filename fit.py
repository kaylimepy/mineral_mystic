import os
import tensorflow
from crystal_vision import CrystalVision


def check_image_formats(directory, allowed_formats=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
    '''
    Check if all image files in a directory and its subdirectories are in the allowed formats.
    
    Parameters:
        directory (str): The directory path to check
        allowed_formats (list): List of allowed file extensions
        
    Returns:
        bool: True if all files are in allowed formats, False otherwise
        list: List of files that are not in the allowed formats
    '''
    invalid_files = []
    
    # Walk through directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Get file extension
            _, ext = os.path.splitext(file)
            
            # Check if extension is in allowed formats
            if ext.lower() not in allowed_formats:
                invalid_files.append(os.path.join(root, file))
                
    return len(invalid_files) == 0, invalid_files


def load_datasets(data_path, batch_size, img_size):
    '''
    Load and return train and validation datasets.
    
    Parameters:
        data_path (str): Path to the data directory
        batch_size (int): Batch size for loading data
        img_size (tuple): Image dimensions (width, height)
        
    Returns:
        tuple: train_dataset, validation_dataset, test_dataset
    '''
    train_dataset = tensorflow.keras.utils.image_dataset_from_directory(
        data_path,
        shuffle=True,
        validation_split=0.2,
        subset='training',
        seed=99,
        batch_size=batch_size,
        image_size=img_size
    )
    
    validation_dataset = tensorflow.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset='validation',
        shuffle=True,
        seed=99,
        batch_size=batch_size,
        image_size=img_size
    )
    
    return train_dataset, validation_dataset


def apply_data_augmentation(dataset):
    '''
    Apply data augmentation to the given dataset.
    
    Parameters:
        dataset (tensorflow.data.Dataset): The dataset to augment
        
    Returns:
        tensorflow.data.Dataset: Augmented dataset
    '''
    data_augmentation = tensorflow.keras.Sequential([
        tensorflow.keras.layers.RandomFlip('horizontal'),
        tensorflow.keras.layers.RandomRotation(0.4),
        tensorflow.keras.layers.RandomContrast(0.2),
        tensorflow.keras.layers.RandomZoom(.5, .2),
        tensorflow.keras.layers.GaussianNoise(0.1)
    ])
    
    return dataset.map(lambda x, y: (data_augmentation(x, training=True), y))


def split_validation_set(validation_dataset):
    '''
    Split the validation dataset into validation and test datasets.
    
    Parameters:
        validation_dataset (tensorflow.data.Dataset): Original validation dataset
        
    Returns:
        tuple: new_validation_dataset, test_dataset
    '''
    validation_batches     = tensorflow.data.experimental.cardinality(validation_dataset)
    test_dataset           = validation_dataset.take(validation_batches // 2)
    new_validation_dataset = validation_dataset.skip(validation_batches // 2)
    
    return new_validation_dataset, test_dataset


if __name__ == "__main__":
    DATA_PATH  = 'temp/minet2'
    BATCH_SIZE = 64
    IMG_SIZE   = (256, 256)

    all_valid, invalid_files = check_image_formats('temp/minet2')
    if not all_valid:
        print(f"Found {len(invalid_files)} invalid files.")
        print('Invalid files:', invalid_files)
    
    train_dataset, validation_dataset = load_datasets(DATA_PATH, BATCH_SIZE, IMG_SIZE)
    train_dataset                     = apply_data_augmentation(train_dataset)
    validation_dataset, test_dataset  = split_validation_set(validation_dataset)
    
    # Create an instance of the CrystalVision class and fine-tune
    mineral_mystic = CrystalVision()
    mineral_mystic.fine_tune(100)
    
    # Recompile the model
    optimizer = tensorflow.keras.optimizers.Nadam(learning_rate=0.001)
    loss      = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics   = ['accuracy']

    mineral_mystic.compile_model(optimizer, loss, metrics)
    
    # Train the model (assuming train_data and val_data are defined or replace with actual data variables)
    mineral_mystic.model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
