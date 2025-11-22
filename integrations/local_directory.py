from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
'''
langchain_community is a package that allows your langchain to interact with other apps.
'''

# Sample documents - simple company knowledge base
documents = [
    "Acme Corp is a technology company founded in 2020. We specialize in AI-powered tools for small businesses. Headquarters is in San Francisco.",
    "Our flagship product is TaskMaster Pro, a project management tool. We also offer DataSync for data integration.",
    "Sarah Chen is our CTO and leads 15 engineers. Mike Rodriguez manages the 8-person sales team.",
    "Employees get 20 days paid vacation per year. We allow 3 remote days per week and provide $2000 learning budgets.",
    "In Q3 2024, Acme raised $5 million in Series A funding. TaskMaster Pro has 10,000 active users."
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.create_documents(documents)
# for doc in texts:
#     print("DOC:", doc.page_content)

embeddings = OllamaEmbeddings(model="embeddinggemma")

vector_db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")

# quick test query
results = vector_db.similarity_search("who's the CTO and how many does he lead?", k=2)
print(f"Result: {results}")