#  Step 1: Load Raw Pdf (S)

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

data_path = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data_path)
print("Lenght of documents",len(documents))

# Step 2 : Create Chunks
def create_chunks(extrected_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunck=create_chunks(extrected_text=documents)
print("Lenght of text chunks",len(text_chunck))

# Step 3 : Create Vector Embeddings
def get_embedding_model():
    embedding_node = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_node

embedding_model=get_embedding_model()

# Step 4 :Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunck,embedding_model)
db.save_local(DB_FAISS_PATH)
  