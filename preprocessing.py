import os
import cv2
import imgaug
import imgaug.augmenters
from PIL import Image
from sklearn.model_selection import train_test_split


def load_and_clean_data(directory):
    """
    Load and clean the image data.

    Parameters:
    - directory (str): Path to the directory containing class subdirectories.

    Returns:
    - Tuple of images and labels.
    """
    classes        = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    images, labels = [], []

    for label, class_name in enumerate(classes):
        class_path = os.path.join(directory, class_name)

        for img_name in os.listdir(class_path):
            if img_name.endswith(('.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path)
                    img.verify()

                    img = cv2.imread(img_path)
                    images.append(img)
                    labels.append(label)

                except Image.UnidentifiedImageError:
                    print(f"Corrupted image detected: {img_path}")
                    os.remove(img_path)
                except Exception as e:
                    print(f"An error occurred with {img_path}: {e}")

    return images, labels


def resize_images(images, target_size=(128, 128)):
    """
    Resize a list of images.

    Parameters:
    - images (list): List of images to resize.
    - target_size (tuple): Desired size (width, height).

    Returns:
    - List of resized images.
    """
    return [cv2.resize(img, target_size) for img in images]


def augment_images(images):
    """
    Apply augmentation to images.

    Parameters:
    - images (list): List of images to augment.

    Returns:
    - List of augmented images.
    """
    seq = imgaug.augmenters.Sequential([
            imgaug.augmenters.Fliplr(0.5), 
            imgaug.augmenters.Crop(percent=(0, 0.1)), 
            imgaug.augmenters.Sometimes(0.5,
                imgaug.augmenters.GaussianBlur(sigma=(0, 0.5))
            ),
            imgaug.augmenters.LinearContrast((0.75, 1.5)),
            imgaug.augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), 
            imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2),
            imgaug.augmenters.Affine(
                scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}, 
                rotate=(-25, 25),
                shear=(-8, 8) 
            )
        ], random_order=True)

    return seq(images=images)


def split_data(images, labels):
    """
    Split data into training, validation, and test sets.

    Parameters:
    - images (list): Images to split.
    - labels (list): Corresponding labels.

    Returns:
    - Tuple of train, validation, and test sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_processed_data(directory='mineral_classes'):
    """
    Process the images and labels from the given directory.

    Parameters:
    - directory (str): Path to the directory containing class subdirectories.

    Returns:
    - Tuple of train, validation, and test sets.
    """
    all_images, labels = load_and_clean_data(directory)
    all_images         = resize_images(all_images)
    augmented_images   = augment_images(all_images)
    return split_data(augmented_images, labels)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_processed_data()

    print('Preprocessing complete!')

