import os
import logging
import tensorflow
import imghdr
import matplotlib.pyplot
from pathlib import Path
from PIL import Image


def remove_invalid_and_corrupted_files(directory, allowed_formats=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
    '''
    Remove files from a directory and its subdirectories that are not in the allowed formats or are corrupted.
    Also converts unsupported but valid image types to JPEG.
    
    Parameters:
        directory (str): The directory path to check
        allowed_formats (list): List of allowed file extensions
    
    Returns:
        int: Number of files removed
    '''
    files_removed = 0

    # Image types accepted by TensorFlow
    img_type_accepted_by_tf = [format[1:] for format in allowed_formats]
    
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = Path(root) / file
            
            # Get file extension
            ext = filepath.suffix
            
            # Check if the file extension is in the allowed formats
            if ext.lower() not in allowed_formats:
                os.remove(filepath)
                files_removed += 1
            
            # Check if the file is a valid image using imghdr
            elif imghdr.what(filepath) is None:
                os.remove(filepath)
                files_removed += 1
                
            # Check if the image type is accepted by TensorFlow, if not convert it to JPEG
            elif imghdr.what(filepath) not in img_type_accepted_by_tf:
                im = Image.open(filepath).convert('RGB')
                im.save(filepath, 'jpeg')

    logging.info(f"Removed {files_removed} invalid or corrupted files.")


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


def get_class_weights(train_dataset):
    '''
    Get class weights for the given dataset.
    
    Parameters:
        train_dataset (tensorflow.data.Dataset): The dataset to get class weights for
        
    Returns:
        dict: Class weights
    '''
    class_weights = {}
    for _, labels in train_dataset:
        for label in labels:
            if label.numpy() not in class_weights:
                class_weights[label.numpy()] = 1
            else:
                class_weights[label.numpy()] += 1
                
    total = sum(class_weights.values())

    for key in class_weights.keys():
        class_weights[key] = round(total / class_weights[key], 2)
        
    return class_weights


def get_datasets(data_path, batch_size, img_size):
    '''
    Get train, validation and test datasets.

    Parameters:
        data_path (str): Path to the data directory
        batch_size (int): Batch size for loading data
        img_size (tuple): Image dimensions (width, height)

    Returns:
        tuple: train_dataset, validation_dataset, test_dataset
    '''
    train_dataset, validation_dataset = load_datasets(data_path, batch_size, img_size)
    validation_dataset, test_dataset  = split_validation_set(validation_dataset)
    
    train_dataset = apply_data_augmentation(train_dataset)

    return train_dataset, validation_dataset, test_dataset


def plot_training_history(history, model_name, metric='accuracy', directory_path='/mnt/data'):
    '''
    Plots the training history of a model and saves the plot to a file.

    Parameters:
        history (History object): The training history as returned by model.fit().
        model_name (str): The name of the model, used for the plot title and filename.
        metric (str): The metric to plot. Default is 'accuracy'.
        save_path (str): The directory where to save the plot. Default is '/mnt/data'.
        
    Returns:
        str: The path where the plot is saved.
    '''
    matplotlib.pyplot.figure(figsize=(10, 6))
    
    matplotlib.pyplot.plot(history.history[metric], label=f"Training {metric.capitalize()}")
    validation_metric = 'val_' + metric

    if validation_metric in history.history:
        matplotlib.pyplot.plot(history.history[validation_metric], label=f"Validation {metric.capitalize()}")

    matplotlib.pyplot.title(f"{model_name} {metric.capitalize()} Over Epochs")
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel(metric.capitalize())
    matplotlib.pyplot.legend()
    
    plot_file_path = f"{directory_path}/{model_name}_{metric}_plot.png"
    matplotlib.pyplot.savefig(plot_file_path)
    
    return plot_file_path
