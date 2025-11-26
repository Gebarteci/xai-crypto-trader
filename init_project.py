import os

def create_structure():
    # Define the folder hierarchy
    folders = [
        "data",
        "models",
        "notebooks",
        "src",
        ".streamlit" # For custom theme config later
    ]
    
    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        # Add a .gitkeep file so empty folders are tracked by git
        with open(os.path.join(folder, ".gitkeep"), "w") as f:
            f.write("")
        print(f"✅ Created: {folder}/")

    # Create __init__.py in src to make it a package
    with open("src/__init__.py", "w") as f:
        f.write("")
    print("✅ Created: src/__init__.py")

    print("\n🚀 Project structure ready! You can now run 'pip install -r requirements.txt'")

if __name__ == "__main__":
    create_structure()