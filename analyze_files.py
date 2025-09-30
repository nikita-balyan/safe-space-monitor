#!/usr/bin/env python3
"""
Safe Space Monitor - File Analysis and Cleanup
Analyzes existing files and identifies what to keep/delete
"""

import os
import shutil
import fnmatch
from pathlib import Path

def analyze_project_structure():
    """Analyze the current project structure"""
    print("🔍 Analyzing project structure...")
    print("=" * 60)
    
    # Define essential files that MUST exist
    essential_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'templates/',
        'static/',
        'data/'
    ]
    
    print("📁 Current Directory Structure:")
    print_tree('.', max_depth=3)
    
    print("\n✅ ESSENTIAL FILES (Must exist):")
    for file in essential_files:
        exists = os.path.exists(file)
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"  {status}: {file}")
    
    print("\n📊 FILE ANALYSIS:")
    analyze_file_types()
    
    print("\n🗑️  FILES TO CLEAN UP:")
    find_files_to_cleanup()

def print_tree(start_path, max_depth=3, prefix=""):
    """Print directory tree structure"""
    if max_depth < 0:
        return
        
    try:
        items = sorted(os.listdir(start_path))
        for i, item in enumerate(items):
            if item.startswith('.'):  # Skip hidden files
                continue
                
            path = os.path.join(start_path, item)
            is_last = i == len(items) - 1
            
            if os.path.isdir(path):
                print(f"{prefix}{'└── ' if is_last else '├── '}📁 {item}/")
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(path, max_depth - 1, new_prefix)
            else:
                print(f"{prefix}{'└── ' if is_last else '├── '}📄 {item}")
    except PermissionError:
        pass

def analyze_file_types():
    """Analyze different file types in project"""
    file_types = {}
    total_size = 0
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and virtual environments
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', '__pycache__']]
        
        for file in files:
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower() or 'no extension'
            size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            file_types[ext] = file_types.get(ext, 0) + 1
            total_size += size
    
    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} files")
    
    print(f"  Total estimated size: {total_size / 1024 / 1024:.2f} MB")

def find_files_to_cleanup():
    """Find files that should be cleaned up"""
    cleanup_patterns = [
        '__pycache__',
        '*.pyc', '*.pyo', '*.pyd',
        '*.db', '*.sqlite3',
        'htmlcov/', '.coverage', '.pytest_cache',
        '.vscode/', '.idea/', '*.swp', '*.swo',
        '.DS_Store', 'Thumbs.db',
        'build/', 'dist/', '*.egg-info',
        'venv/', 'env/'
    ]
    
    cleanup_files = []
    
    for pattern in cleanup_patterns:
        if pattern.endswith('/'):  # Directory pattern
            dir_pattern = pattern.rstrip('/')
            if os.path.exists(dir_pattern):
                size = get_size(dir_pattern)
                cleanup_files.append((dir_pattern, size, 'directory'))
        else:  # File pattern
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        full_path = os.path.join(root, file)
                        size = get_size(full_path)
                        cleanup_files.append((full_path, size, 'file'))
    
    for path, size, file_type in sorted(set(cleanup_files)):
        print(f"  🗑️  {path} ({size}) - {file_type}")

def get_size(path):
    """Get human-readable size of file/directory"""
    if os.path.isfile(path):
        size = os.path.getsize(path)
    else:
        size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                  for dirpath, dirnames, filenames in os.walk(path) 
                  for filename in filenames)
    
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size/1024:.1f} KB"
    else:
        return f"{size/1024/1024:.1f} MB"

def create_cleanup_script():
    """Create a cleanup script based on analysis"""
    print("\n🛠️  Creating cleanup script...")
    
    cleanup_script = """#!/bin/bash
# Safe Space Monitor - Cleanup Script
# Removes unnecessary files for deployment

echo "🧹 Starting Safe Space Monitor cleanup..."

# Remove Python cache files
echo "Removing Python cache files..."
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove test and coverage files
echo "Removing test coverage files..."
rm -rf htmlcov/ .coverage .pytest_cache/

# Remove build and distribution files
echo "Removing build files..."
rm -rf build/ dist/ *.egg-info/ .eggs/ eggs/

# Remove IDE files
echo "Removing IDE files..."
rm -rf .vscode/ .idea/ *.swp *.swo

# Remove OS files
echo "Removing OS files..."
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true

# Remove old Replit files (keep current ones)
echo "Cleaning old files..."
rm -f flask_pid_*.txt
rm -f start_*.sh
rm -f uv_*.lock

echo "✅ Cleanup complete!"
echo ""
echo "📁 Remaining important files:"
find . -name "*.py" -type f | grep -v "__pycache__" | head -10
echo ""
echo "💾 Total size after cleanup:"
du -sh . 2>/dev/null || echo "Run 'dir' on Windows to see size"
"""

    with open('cleanup.sh', 'w') as f:
        f.write(cleanup_script)
    
    os.chmod('cleanup.sh', 0o755)  # Make executable
    print("✅ Created cleanup.sh script")

def main():
    """Main analysis function"""
    print("🚀 Safe Space Monitor - File Analysis")
    print("=" * 60)
    
    analyze_project_structure()
    create_cleanup_script()
    
    print("\n" + "=" * 60)
    print("🎯 CLEANUP INSTRUCTIONS:")
    print("1. Run: bash cleanup.sh")
    print("2. Or run these commands manually:")
    print("   find . -name '__pycache__' -type d -exec rm -rf {} +")
    print("   find . -name '*.pyc' -delete")
    print("   rm -rf htmlcov/ .pytest_cache/ build/ dist/")
    print("")
    print("📋 FILES TO KEEP:")
    print("   ✅ app.py, routes.py, database.py")
    print("   ✅ templates/, static/, data/")
    print("   ✅ requirements.txt, README.md")
    print("   ✅ models/ (contains your ML model)")
    print("")
    print("🚀 After cleanup, run: git add . && git commit -m 'Clean project'")

if __name__ == "__main__":
    main()