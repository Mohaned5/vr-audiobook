import os
from PIL import Image

def check_images(dataset_dir):
    # Path to your dataset directory
    data_dir = os.path.join(dataset_dir, 'train')  # Change to your train directory

    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} does not exist.")
        return
    
    # List all images in the train directory
    all_files = os.listdir(data_dir)
    image_files = [f for f in all_files if f.endswith('.png') or f.endswith('.jpg')]  # Add other formats if necessary

    corrupted_files = []
    missing_files = []

    # Loop over all image files
    for image_file in image_files:
        file_path = os.path.join(data_dir, image_file)

        try:
            # Try to open the image
            with Image.open(file_path) as img:
                img.verify()  # Verify the image
            print(f"File {image_file} is valid.")
        
        except (OSError, IOError):
            print(f"Corrupted file: {file_path}")
            corrupted_files.append(file_path)
        
        # Check if file is missing or empty
        if os.stat(file_path).st_size == 0:
            print(f"Empty file found: {file_path}")
            missing_files.append(file_path)

    # Report
    if corrupted_files:
        print(f"\nTotal corrupted files: {len(corrupted_files)}")
        print("Corrupted files:", corrupted_files)
    else:
        print("\nNo corrupted files found.")
    
    if missing_files:
        print(f"\nTotal missing/empty files: {len(missing_files)}")
        print("Missing files:", missing_files)
    else:
        print("\nNo missing files found.")

if __name__ == "__main__":
    dataset_directory = 'data/Panime'  # Set this to your dataset path
    check_images(dataset_directory)
