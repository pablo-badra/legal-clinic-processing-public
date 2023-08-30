import os

# Specify the folder containing the .csv files. replace this as necessary
folder_path = "/Users/stjames/Dropbox/Legal/Clinic - GSA Remote/Shared Remote Intake Folder"

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a .csv file
    if filename.endswith(".pdf"):
        # Replace blank spaces, commas, dashes, and parentheses with underscores in the filename
        new_filename = filename.replace(" ", "_")
        new_filename = new_filename.replace(",", "_")
        new_filename = new_filename.replace("-", "_")
        new_filename = new_filename.replace("(", "_")
        new_filename = new_filename.replace(")", "_")
        
        # Create the full file paths for the old and new filenames
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
