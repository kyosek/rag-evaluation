import os
from datetime import datetime
import re

def rename_files(directory_path, old_date='20241001', new_date='20241030'):
    """
    Rename files by updating the date portion in the filename.
    
    Args:
        directory_path (str): Path to the directory containing files
        old_date (str): Date string to replace (format: YYYYMMDD)
        new_date (str): New date string (format: YYYYMMDD)
    """
    # Validate date format
    date_pattern = r'\d{8}'
    if not (re.match(date_pattern, old_date) and re.match(date_pattern, new_date)):
        raise ValueError("Dates must be in YYYYMMDD format")

    # List all files in directory
    files_renamed = 0
    
    for filename in os.listdir(directory_path):
        if old_date in filename:
            new_filename = filename.replace(old_date, new_date)
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)
            
            try:
                os.rename(old_filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")
                files_renamed += 1
            except OSError as e:
                print(f"Error renaming {filename}: {e}")
    
    print(f"\nTotal files renamed: {files_renamed}")

# Example usage
if __name__ == "__main__":
    # Replace with your directory path
    directory = "Data/Arxiv/RawExamData"
    old_date = "2024103016"
    new_date = "2024103108"
    
    try:
        rename_files(directory, old_date, new_date)
    except Exception as e:
        print(f"An error occurred: {e}")
