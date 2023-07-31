import os
import pinecone
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain

pdfs = ["CG-MED-23.pdf", "CG-SURG-95.pdf"]

data = []
for pdf in pdfs:
    loader = PyPDFLoader(f"{pdf}")
    _pdf = loader.load()
    data = data + _pdf
print("Total Length of data: ", len(data))

page_info = {}
for pdf in data:
    if [f for f in pdfs if pdf.metadata['source'].split("/")[-1] in f]:
        if pdf.metadata['source'].split("/")[-1] in page_info:
            page_info[pdf.metadata['source'].split("/")[-1]] += 1
        else:
            page_info[pdf.metadata['source'].split("/")[-1]] = 1
print("Total pages in each pdf: ", page_info)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(data)
print(len(docs))

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_tPijqvaCKVoSwscgcqvUMLLLcrchBzSXXQK"
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') ## LLamma2 generates embeddings of 384 dimensions

index_name = 'nlpandavas'

pinecone.init(
    api_key="ef6dd11e-91a1-4440-9989-bdfd4321fa6f",
    environment="us-west4-gcp-free"
)

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=384  # 384 dim of all-MiniLM-L6-v2
    )

index = pinecone.GRPCIndex(index_name)
index.describe_index_stats()

texts = [t.page_content for t in docs]
metadatas = [t.metadata for t in docs]

docsearch=Pinecone.from_texts(texts=texts, embedding=embeddings, index_name=index_name, metadatas=metadatas)

index.describe_index_stats()

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(
    model_name = 'gpt-3.5-turbo',
    temperature=0.0
)

def get_comment(policy: str):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":10, "filter": {"source" : policy + ".pdf"}})
    )
    return qa

# ### Inferencing Part!!
# policy = 'CG-SURG-95' ## Should come from UI

# out_llm_qa = get_comment(policy=policy)

# query = f"What are the medical necessary conditions associated with {policy}?" ## Should come from UI
# output = out_llm_qa.run(query) ## Output for UI

def output_llm(policy,prompt):
    llm_qa = get_comment(policy=policy)
    output = llm_qa.run(prompt)
    return output