import nltk

def download_nltk_data():
    nltk_resources = [
        'stopwords', 
        'punkt', 
        'wordnet', 
        'omw-1.4'
    ]
    
    for resource in nltk_resources:
        try:
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

if __name__ == "__main__":
    download_nltk_data()
