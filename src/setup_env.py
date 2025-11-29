"""
Setup script for ControlGym Mass-Spring-Damper Project.

Checks environment, installs dependencies, and generates lock file.
"""

import sys
import subprocess
import os
import pkg_resources

def check_python_version():
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("WARNING: Python 3.8+ is recommended.")

def install_dependencies():
    print("\nInstalling dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def generate_lock_file():
    print("\nGenerating requirements.lock...")
    try:
        # Use pip freeze to generate lock file
        result = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        with open("requirements.lock", "wb") as f:
            f.write(result)
        print("requirements.lock generated.")
    except subprocess.CalledProcessError as e:
        print(f"Error generating lock file: {e}")

def main():
    print("="*50)
    print("ControlGym Project Setup")
    print("="*50)
    
    check_python_version()
    
    if os.path.exists("requirements.txt"):
        install_dependencies()
        generate_lock_file()
    else:
        print("Error: requirements.txt not found!")
        sys.exit(1)
        
    print("\nSetup complete! You can now run the training scripts.")
    print("Example: python src/train_ppo_msd.py")

if __name__ == "__main__":
    main()
