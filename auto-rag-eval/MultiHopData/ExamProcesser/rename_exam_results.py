import os
import logging
from pathlib import Path
from typing import List, Optional

def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_json_files(directory: str) -> List[Path]:
    """
    Get all JSON files in the specified directory.
    
    Args:
        directory (str): Directory path to search for JSON files
        
    Returns:
        List[Path]: List of paths to JSON files
    """
    return list(Path(directory).glob("*.json"))

def rename_files(directory: str, dry_run: bool = False) -> None:
    """
    Rename all JSON files in the directory by replacing hyphens with underscores.
    
    Args:
        directory (str): Directory containing the files to rename
        dry_run (bool): If True, only show what would be done without actually renaming
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Get all JSON files
        json_files = get_json_files(directory)
        
        if not json_files:
            logger.warning(f"No JSON files found in directory: {directory}")
            return
            
        renamed_count = 0
        for file_path in json_files:
            if "-" in file_path.name:
                new_name = file_path.name.replace("-", "_")
                new_path = file_path.parent / new_name
                
                if dry_run:
                    logger.info(f"Would rename: {file_path.name} -> {new_name}")
                else:
                    try:
                        file_path.rename(new_path)
                        logger.info(f"Renamed: {file_path.name} -> {new_name}")
                        renamed_count += 1
                    except OSError as e:
                        logger.error(f"Error renaming {file_path.name}: {str(e)}")
                        
        if not dry_run:
            logger.info(f"Successfully renamed {renamed_count} files")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    
    for task in task_domains:
        rename_files(f"auto-rag-eval/MultiHopData/{task}/exam_results")
