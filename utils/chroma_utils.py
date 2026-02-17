import logging
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from config import settings
from PyPDF2 import PdfReader
from io import BytesIO

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Setup logging
logging.basicConfig(filename=settings.log_path, level=logging.INFO)

def get_chroma_db(collection_name: str = "zendalona"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",

        google_api_key=settings.gemini_api_key
    )
    return Chroma(
        collection_name=collection_name,
        persist_directory=settings.chroma_db_path,
        embedding_function=embeddings,
        
    )

def index_documents_to_chroma(documents: list[Document], collection_name: str = "zendalona", force_reindex: bool = False) -> int:
    try:
        db = get_chroma_db(collection_name)
        
        if force_reindex:
            # If force reindexing, add all documents without checking for duplicates
            db.add_documents(documents)
            logging.info(f"Force indexed {len(documents)} documents to ChromaDB collection '{collection_name}'")
            return len(documents)
        else:
            # Get existing document URLs/sources to avoid duplicates
            existing_docs = db.get(include=["metadatas"])
            existing_sources = {doc["source"] for doc in existing_docs["metadatas"] if "source" in doc}
            
            # Filter out documents whose sources are already indexed
            new_documents = [doc for doc in documents if doc.metadata.get("source") not in existing_sources]
            
            if not new_documents:
                logging.info("No new documents to index; all sources already exist in ChromaDB")
                return 0
            
            # Add new documents to ChromaDB
            db.add_documents(new_documents)
            logging.info(f"Indexed {len(new_documents)} new documents to ChromaDB collection '{collection_name}'")
            return len(new_documents)
    except Exception as e:
        logging.error(f"Error indexing documents: {str(e)}")
        raise

def process_pdf(file: BytesIO, filename: str) -> list[Document]:
    try:
        # Read PDF content
        pdf_reader = PdfReader(file)
        documents = []
        
        # Extract text from each page
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "type": "pdf"
                    }
                )
                documents.append(doc)
        
        if not documents:
            logging.warning(f"No text extracted from PDF: {filename}")
            return []
        
        logging.info(f"Extracted {len(documents)} pages from PDF: {filename}")
        return documents
    except Exception as e:
        logging.error(f"Error processing PDF {filename}: {str(e)}")
        return []

def process_and_index_file(file_content: bytes, filename: str) -> list[Document]:
    if filename.endswith(".pdf"):
        return process_pdf(BytesIO(file_content), filename)
    # Add other file types here
    # elif filename.endswith(".txt"):
    #     return process_txt(BytesIO(file_content), filename)
    else:
        logging.warning(f"File type not supported for {filename}")
        return []

def update_document(document_id: str, document: Document, collection_name: str = "zendalona"):
    try:
        db = get_chroma_db(collection_name)
        db.update_document(document_id, document)
        logging.info(f"Successfully updated document {document_id} in collection {collection_name}")
        return True
    except Exception as e:
        logging.error(f"Error updating document {document_id} in collection {collection_name}: {str(e)}")
        return False

def list_collections():
    try:
        db = get_chroma_db()
        collections = db._client.list_collections()
        return [collection.name for collection in collections]
    except Exception as e:
        logging.error(f"Error listing collections: {str(e)}")
        return []

def delete_collection(collection_name: str):
    try:
        db = get_chroma_db()
        db._client.delete_collection(name=collection_name)
        logging.info(f"Successfully deleted collection: {collection_name}")
        return True
    except Exception as e:
        logging.error(f"Error deleting collection {collection_name}: {str(e)}")
        return False

def get_collection_documents(collection_name: str):
    try:
        db = get_chroma_db(collection_name)
        results = db.get(include=["metadatas"])
        
        # Extracting id and source from the results
        documents = []
        for i, doc_id in enumerate(results['ids']):
            source = results['metadatas'][i].get('source', 'N/A')
            documents.append({'id': doc_id, 'source': source})
            
        return documents
    except Exception as e:
        logging.error(f"Error getting documents from collection {collection_name}: {str(e)}")
        return None

def delete_document_from_collection(collection_name: str, document_id: str):
    try:
        db = get_chroma_db(collection_name)
        db.delete(ids=[document_id])
        logging.info(f"Successfully deleted document {document_id} from collection {collection_name}")
        return True
    except Exception as e:
        logging.error(f"Error deleting document {document_id} from collection {collection_name}: {str(e)}")
        return False