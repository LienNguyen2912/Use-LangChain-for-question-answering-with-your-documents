from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
import os
from dotenv import load_dotenv
import re

load_dotenv()   # Load GOOGLE_API_KEY from .env file
DB_DIR_PATH = "./DB/"
SEARCH_TYPE_SIMILARITY = 0  # default search type
SEARCH_TYPE_MAX_MARGINAL_RELEVANCE = 1

def create_vector_db(embedding, filename):
    file_path = "./csv/" + filename +".csv"
    # specify column "question" for vectorizing
    loader = CSVLoader(file_path = file_path, source_column="question")
    db_file_path = "./DB/" + filename
    # 
    if check_directory_size(db_file_path) == True:
        print('db file is existed')
        return db_file_path

    # create vectorized db    
    data = loader.load()    
    """
    #type of data is <class 'list'>, type of data[0] is <class 'langchain.schema.document.Document'>
    example of a Document item:
    page_content="url: http://www.freebase.com/view/en/jamaica\nquestion: what does jamaican people speak?\nanswers: ['Jamaican Creole English Language' 'Jamaican English']"
    metadata={'source': 'what does jamaican people speak?', 'row': 0}
    """

    vectorDB = FAISS.from_documents(documents = data,embedding = embedding)
    vectorDB.save_local(db_file_path)
    return db_file_path

def build_qa_chain(search_type = 0):
    # Initialize
    embedding = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-large")
    # debug >>
    # e = embedding.embed_query("book")
    # print(e[:10])
    # print(len(e))  # 768
    # debug <<
    print("###############")
    db_file_path = create_vector_db(embedding, "mytest")
    vectorDB = FAISS.load_local(db_file_path,embeddings = embedding)
    if search_type !=0 :
        # Fetch more documents for the MMR algorithm to consider, but only return the top 5
        retrieve = vectorDB.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 50})
    else:
        # Only retrieve documents that have a relevance score above a certain threshold
        retrieve = vectorDB.as_retriever(search_type = "similarity_score_threshold", search_kwargs ={'score_threshold' : 0.6})  
    # debug >>
    # rdocs = retrieve.get_relevant_documents("what did james k polk do before he was president?")
    # print("get_relevant_documents are:", rdocs)
    # debug <<
    # create prompt template with input as context and question
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:{context}
    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm= GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1), 
        chain_type="stuff", 
        retriever=retrieve,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs)

    return qa

def getAnswer(query):
    qa = build_qa_chain(SEARCH_TYPE_MAX_MARGINAL_RELEVANCE) # SEARCH_TYPE_SIMILARITY
    result = qa({"query": query})
    answer = result["result"]
    context = result["source_documents"]
    return answer, context

def extract_question_part(data):
    # an example of data:
    # "url: http://www.freebase.com/view/en/jamaica\nquestion: what does jamaican people speak?\nanswers: ['Jamaican Creole English Language' 'Jamaican English']"
    result = []
    for item in data:
        # Use a regular expression to extract the content between "\nquestion: " and "\nanswers:"
        match = re.search(r'\nquestion: (.*?)\nanswers:', item.page_content)
        if match:
            question = match.group(1)
            item.page_content = question
            result.append(item)
    return result

def check_directory_size(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # Check if the directory has files
        if any(os.scandir(directory_path)):
            # Calculate the total size of the directory
            total_size = sum(file.stat().st_size for file in os.scandir(directory_path) if file.is_file())
            
            if total_size > 0:
                return True
            else:
                return False
        else:
            return False
    else:
        return False

if __name__ == "__main__":
    answer, context = getAnswer("where did edgar allan poe died?")
    print("#####Answer: ")
    print(answer)
    print("#####Context:")
    print(context)
