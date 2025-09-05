#!/usr/bin/env python3
"""
Script to fix folder structure
"""

import os
import shutil
from pathlib import Path

def fix_folder_structure():
    """Fix nested src folders"""
    base_dir = Path("D:/CODE_PLAYGROUND/SensoryCalm")
    
    # Check if we have nested src folders
    nested_src = base_dir / "src" / "src" / "src"
    
    if nested_src.exists():
        print("Found nested src folders. Fixing...")
        
        # Move files from the deepest src folder to the main src folder
        main_src = base_dir / "src"
        
        # Create main src folder if it doesn't exist
        main_src.mkdir(exist_ok=True)
        
        # Move all files from nested src to main src
        for item in nested_src.iterdir():
            if item.is_file():
                shutil.move(str(item), str(main_src / item.name))
                print(f"Moved: {item.name}")
        
        # Remove empty nested folders
        try:
            (base_dir / "src" / "src").rmdir()  # Remove middle src folder
            nested_src.rmdir()  # Remove deepest src folder
            print("Removed empty nested folders")
        except:
            print("Could not remove nested folders (they might not be empty)")
        
        print("Folder structure fixed!")
    else:
        print("No nested src folders found.")

if __name__ == "__main__":
    fix_folder_structure()