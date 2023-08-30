from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.sitemap import SitemapLoader
from  langchain.schema import Document
import json
from typing import Iterable

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

# Create vector database
def create_vector_db(site, file):
    sitemap_loader = SitemapLoader(web_path=site)
    sitemap_loader.requests_per_second = 6
    
    # Optional: avoid `[SSL: CERTIFICATE_VERIFY_FAILED]` issue
    sitemap_loader.requests_kwargs = {"verify": True}

    documents = sitemap_loader.load()
    
    save_docs_to_jsonl(documents,file)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=80)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    sitemap ='https://academy.binance.com/article/sitemap_article_en.xml'
    file = 'crypto_data.jsonl'
    create_vector_db(sitemap, file)

