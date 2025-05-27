# Medical Chatbot

A conversational AI system designed to answer medical queries using Retrieval-Augmented Generation (RAG) with FAISS vector database and Hugging Face language models.

## 🏥 Project Overview

This Medical Chatbot leverages advanced NLP techniques to provide accurate medical information by:
- Using FAISS vector database for efficient document retrieval
- Implementing RAG (Retrieval-Augmented Generation) architecture
- Utilizing Hugging Face's Mistral-7B-Instruct model for response generation
- Providing context-aware answers from medical knowledge base

## 🚀 Features

- **Intelligent Document Retrieval**: Uses FAISS for fast similarity search
- **Context-Aware Responses**: Provides answers based on retrieved medical documents
- **Customizable Prompts**: Tailored prompt templates for medical queries
- **Source Attribution**: Shows source documents for transparency
- **Environment Configuration**: Secure API token management

## 📁 Project Structure

```
Medical-Chatbot/
├── connect_memory_with_llm.py    # Main application file
├── check_requirements.py        # Dependencies checker
├── data/                        # Medical documents directory
├── vectorstore/                 # FAISS vector database
│   └── db_faiss/
├── .env                         # Environment variables (not tracked)
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
└── requirements.txt             # Project dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Medical-Chatbot
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
# Copy the example file
cp .env.example .env
# Edit .env and add your Hugging Face token
```

5. **Verify installation:**
```bash
python check_requirements.py
```

## 🔑 Configuration

### Environment Variables

Create a `.env` file with the following:

```env
HF_TOKEN="your_hugging_face_token_here"
```

### Getting Hugging Face Token

1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Select "Read" access
4. Copy the generated token to your `.env` file

## 📚 Dependencies

```txt
langchain
langchain-core
langchain-community
langchain-huggingface
python-dotenv
faiss-cpu
huggingface-hub
sentence-transformers
```

## 🚀 Usage

### Running the Chatbot

```bash
python connect_memory_with_llm.py
```

### Example Interaction

```
Write Query Here: What are the symptoms of diabetes?
RESULT: Based on the medical documents, common symptoms of diabetes include...
SOURCE DOCUMENTS: [Document sources will be displayed here]
```

### Key Components

#### 1. Vector Database Setup
```python
# Load FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
```

#### 2. Language Model Configuration
```python
# Setup Mistral LLM
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        token=HF_TOKEN
    )
    return llm
```

#### 3. Custom Prompt Template
```python
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
```

## 🔧 Technical Details

### Architecture
- **Retrieval System**: FAISS vector database with sentence-transformers embeddings
- **Language Model**: Mistral-7B-Instruct-v0.3 via Hugging Face Inference API
- **Framework**: LangChain for orchestrating RAG pipeline
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2

### Key Features
- **Document Retrieval**: Top-k similarity search (k=3)
- **Response Generation**: Context-aware medical responses
- **Source Attribution**: Returns source documents for verification
- **Error Handling**: Graceful handling of API errors

## 🔒 Security Considerations

- **.env File**: Never commit API tokens to version control
- **Gitignore**: Properly configured to exclude sensitive files
- **Token Management**: Secure storage of Hugging Face API tokens
- **Deserialization**: Uses `allow_dangerous_deserialization=True` for FAISS (review for production)

## ⚠️ Important Notes

### Medical Disclaimer
This chatbot is for informational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical decisions.

### Data Requirements
- Ensure medical documents are properly formatted in the `data/` directory
- Vector database must be pre-built using document embeddings
- Regular updates to medical knowledge base recommended

## 🐛 Troubleshooting

### Common Issues

1. **Token Errors**: Ensure HF_TOKEN is correctly set in `.env` file
2. **FAISS Loading**: Verify vector database exists in `vectorstore/db_faiss/`
3. **Dependencies**: Run `python check_requirements.py` to verify installation

### Error Solutions

```bash
# If FAISS index not found:
# Ensure vectorstore directory exists and contains proper index files

# If token authentication fails:
# Regenerate Hugging Face token and update .env file

# If dependencies missing:
pip install -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for language models and embeddings
- LangChain for RAG framework
- FAISS for efficient vector search
- Sentence Transformers for embeddings

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**⚠️ Medical Disclaimer**: This AI chatbot provides general medical information for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions.
