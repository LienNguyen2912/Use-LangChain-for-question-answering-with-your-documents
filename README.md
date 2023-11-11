# Use LangChain for question answering with your documents
## We will use
  - Langchain + Google Palm + Huggingface instructor embeddings
  - FAISS
  - Streamlit
  - Train and test documents in csv format
## What we're going to build is
![image](https://github.com/LienNguyen2912/Use-LangChain-to-question-answering-with-your-documents/assets/73010204/1a56f66c-51dc-43d7-8ef0-2df92bcc53e6)
## Prepare your documents
CSV files with 3778 rows and 3 columns each, as illustrated below. Additionally, we prepared an other .csv file for testing purposes (just for fun).
![image](https://github.com/LienNguyen2912/Use-LangChain-to-question-answering-with-your-documents/assets/73010204/d6c22953-ae96-440d-9730-9be5bacb76df)</br>

## Create a vector database
### Initialize embeddings using the Hugging Face model
```
embedding = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-large")
```
Let's explore the vector representation of the word `book`.
```
e = embedding.embed_query("book")
print(e[:10])
print(len(e))
```
We 'll get
```
max_seq_length  512
[-0.02265726402401924, 0.0024643256328999996, 0.00616015400737524, 0.01837732270359993, 0.010981856845319271, 0.005622340831905603, -0.018421605229377747, -0.01336435042321682, -0.06461983919143677, 0.021938087418675423]
768
```
### Load your document and create vector DB
The type of `data` is `<class 'list'>`, and the type of `data[i]` is `<class 'langchain.schema.document.Document'>`.</br>
Below represents an example of a `Document` item, which consists of two parts: `page_content` and `metadata.`</br>
```
page_content="url: http://www.freebase.com/view/en/jamaica\nquestion: what does jamaican people speak?\nanswers: ['Jamaican Creole English Language' 'Jamaican English']"
metadata={'source': 'what does jamaican people speak?', 'row': 0}
```
```
loader = CSVLoader(file_path = file_path, source_column="question")
data = loader.load()    
vectorDB = FAISS.from_documents(documents = data,embedding = embedding)
vectorDB.save_local(db_file_path)
```
### Create RetrievalQA chain
There are two options for `search_type`: `MMR` (Max Marginal Relevance) or `similarity_score_threshold`. The default is `similarity_score_threshold`
```
vectorDB = FAISS.load_local(db_file_path,embeddings = embedding)
retrieve = vectorDB.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 50})
# You can try this search_type too
# retrieve = vectorDB.as_retriever(search_type = "similarity_score_threshold", search_kwargs ={'score_threshold' : 0.6})
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
```
You can obtain a free `GOOGLE_API_KEY` from https://makersuite.google.com/app/apikey.</br>
By default, the `retrieval` generates 4 relevant sources. We can modify this using the parameter `k`. In this example, I've set `k` to 3. Let's try it out.
```
rdocs = retrieve.get_relevant_documents("what did james k polk do before he was president?")
print("get_relevant_documents are:", rdocs)
```
![image](https://github.com/LienNguyen2912/Use-LangChain-to-question-answering-with-your-documents/assets/73010204/ed2b8ba5-0c96-4249-9dc2-229fd250671c)</br>
**The result of `get_relevant_documents()` is, in fact, the input `context` for the `PromptTemplate` variable.**
### All set! Okay, ask some questions :D
```
result = qa({"query": query})
answer = result["result"]
context = result["source_documents"]
```
Let's spice things up by throwing in some questions from the `mytest.csv`—a file that hasn't been used to build our vector DB. 
![image](https://github.com/LienNguyen2912/Use-LangChain-to-question-answering-with-your-documents/assets/73010204/9aab200b-818c-48b8-ab79-341215de351b)</br>
The expected outcome recorded in mytest.csv is also "Mobile". Sweet!</br>
![image](https://github.com/LienNguyen2912/Use-LangChain-to-question-answering-with-your-documents/assets/73010204/45923500-75a6-4c36-a2b9-ebb2e046dd4e)</br>
**That's it!**
## How to run:
- Execute `pip install -r requirement.txt` to install necessary libraries.
- Insert your google api key in the .evn file
- Execute streamlit run .\main.py , http://localhost:8501/ will be launched 😆 😋
