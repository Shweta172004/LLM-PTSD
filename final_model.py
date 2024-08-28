import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
from sklearn.cluster import KMeans

DB_FAISS_PATH = "vectorstore/db_faiss"


def get_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

st.title("LLM-PTSD")
st.markdown("<h3 style = 'text-align: center; color = white;'>Built by <a href='https://github.com/Shweta172004'>LLM by Shweta </a></h3>", unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader("Upload your Data", type = "csv")
if uploaded_files:
    with tempfile.NamedTemporaryFile(delete = False) as tmp_file:
        tmp_file.write(uploaded_files.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding = 'utf-8', csv_args = {
        'delimiter':','
    })

    data = loader.load()
    st.json(data)
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {'device' : 'cpu'}
    )  
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)  
    tokenizer, model = get_llm("DataCleaning/Fine-tune-using-LoRA")  #
    chain = ConversationalRetrievalChain.from_llm(llm = model, retriever = db.as_retriever())

    def conversation(query):
        result = chain({"question":query, "chat_history": st.session_state['history']})
        st.session_state['history'].append(query, result["answer"])
        return result['answer']
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello, I am here to help!"] 

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Enter your message"]    

    #container for the chat history
    response_container = st.container()
    container = st.container()
    with container:
        with st.form(key = "my_form", clear_on_submit = True):
            user_input = st.text_input("Query:", placeholder = "Talk to me", key = 'input')
            submit_button = st.form_submit_button(label = 'submit')

        if submit_button and user_input:
            output = conversation(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

            ehr_data_df = pd.DataFrame(data)  # Convert loaded data to a DataFrame
            combined_text = ' '.join(output)  # Combine LLM response text

            # Creating embeddings for clustering
            combined_embeddings = embeddings.embed_text(combined_text)
            ehr_embeddings = ehr_data_df.apply(lambda row: embeddings.embed_text(' '.join(row)), axis=1).tolist()

            # K-Means Clustering
            kmeans = KMeans(n_clusters=2, random_state=0)
            all_embeddings = [combined_embeddings] + ehr_embeddings
            kmeans.fit(all_embeddings)

            # Verification purpose only, no output shown
            llm_cluster = kmeans.predict([combined_embeddings])[0]
            ehr_clusters = kmeans.predict(ehr_embeddings)
            closest_cluster_match = min(ehr_clusters, key=lambda x: abs(x - llm_cluster))    

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i] , is_user = True, key = str[i]+'_user')
                message(st.session_state['generated'][i], key = str[i])                   
