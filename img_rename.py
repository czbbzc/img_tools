import os

def rename_images(source_folder, destination_folder):
    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Iterate over each file
    for id, file in enumerate(files):
        # Check if the file is an image (you can add more image extensions if needed)
        if file.endswith(".jpg") or file.endswith(".png"):
            # Generate a new name for the file
            # new_name = "new_" + file
            new_name = f'main_{id:0>5d}.png'

            # Build the full paths for the source and destination files
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, new_name)

            # Rename and move the file
            os.rename(source_path, destination_path)

# Specify the source and destination folders
source_folder = "20240403mao/mao_45/images_copy"
destination_folder = "20240403mao/mao_45/images_new"

os.makedirs(destination_folder, exist_ok=True)

# Call the function to rename and move the images
rename_images(source_folder, destination_folder)