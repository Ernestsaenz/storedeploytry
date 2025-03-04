
#!/usr/bin/env python
# checkdb.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

def main():
    load_dotenv()
    
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        return
    
    persist_directory = "./chroma_db"
    collection_name = "hepatology_docs"
    embedding_function = OpenAIEmbeddings()
    
    # Check if the directory exists
    if not Path(persist_directory).exists():
        print(f"Error: Vector database directory '{persist_directory}' does not exist")
        return
    
    try:
        # Load the database
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
        )
        
        # Get the document count
        collection = db.get()
        doc_count = len(collection['ids'])
        
        print(f"Number of documents in collection '{collection_name}': {doc_count}")
        
        # Additional information
        if doc_count > 0:
            print(f"Document IDs (first 5): {collection['ids'][:5]}")
            print(f"Metadata example: {collection['metadatas'][0] if collection['metadatas'] else 'None'}")
        
    except Exception as e:
        print(f"Error accessing vector database: {str(e)}")

if __name__ == "__main__":
    main()
