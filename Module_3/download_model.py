from sentence_transformers import SentenceTransformer

def main():
    """
    Downloads and caches the sentence-transformer model.
    Running this script once will save the model to your local cache,
    preventing a download every time the main scoring script runs.
    """
    model_name = 'all-MiniLM-L6-v2'
    print(f"Downloading and caching model: {model_name}")
    
    try:
        SentenceTransformer(model_name)
        print("\nModel downloaded and cached successfully.")
        print("You can now run the main ATS scoring engine.")
    except Exception as e:
        print(f"\nAn error occurred during model download: {e}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
