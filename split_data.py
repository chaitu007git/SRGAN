import random
import os
import glob
import pickle

def split_data():
    """
    This function splits the image data into train and test.

    Args:
        None

    Returns:
        None
    """
    print("[INFO] Loading all images from the dataset...")

    # Define supported image formats
    supported_formats = ['*.png', '*.jpg', '*.jpeg']

    # Collect all images with supported formats
    all_images_list = []
    for ext in supported_formats:
        all_images_list.extend(glob.glob(f"/content/Data_1024/{ext}", recursive=False))

    if not all_images_list:
        print("[ERROR] No images found in the dataset path.")
        print("[DEBUG] Check your dataset path or file formats.")
        return

    print(f"[INFO] Total images found: {len(all_images_list)}")
    
    ## Shuffle the data
    print("[INFO] Shuffling the data...")
    random.shuffle(all_images_list)

    ## Split the data into train and test
    split_index = int(0.9 * len(all_images_list))  # 90% for training
    train_images = all_images_list[:split_index]
    test_images = all_images_list[split_index:]

    print(f"[INFO] Train images: {len(train_images)}")
    print(f"[INFO] Validation images: {len(test_images)}")

    ## Save the train and test split
    with open("/content/train_images.pkl", "wb") as fp:
        pickle.dump(train_images, fp)
        print("[INFO] Train images saved to /content/train_images.pkl")
    
    with open("/content/val_images.pkl", "wb") as fp:
        pickle.dump(test_images, fp)
        print("[INFO] Validation images saved to /content/val_images.pkl")

if __name__ == "__main__":

    ## Create the train and test data
    print("[INFO] Starting data split process...")
    split_data()
    print("[INFO] Data split process completed successfully.")
