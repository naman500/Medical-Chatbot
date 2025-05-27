import pkg_resources
import sys

def check_requirements():
    # Required packages
    required = {
        'langchain',
        'langchain-core',
        'langchain-community',
        'langchain-huggingface',
        'python-dotenv',
        'faiss-cpu',
        'huggingface-hub',
        'sentence-transformers'
    }
    
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        print("Missing packages:")
        for pkg in missing:
            print(f"❌ {pkg}")
        print("\nInstall missing packages using:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ All required packages are installed!")

if __name__ == "__main__":
    check_requirements()