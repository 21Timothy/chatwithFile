from flask import Flask, request, jsonify, g, render_template
from flask_cors import CORS
import openai
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain #VectorDBQA, 
# from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "80a0f8771352f739cc41e5b9eaf15e9c039c820bb9d397fa3370deb5c5cdd0e5"

app = Flask(__name__)
cors = CORS(app)



qa = ''
chat_history = []
@app.route('/api/upload', methods=['POST'])
def upload():
    global qa
    try:
        file = request.files.get('file')
        save_path = './file_folder/'+file.filename
        file.save(save_path)
        extension = os.path.splitext(file.filename)[1]
        if extension in [".csv", ".xls", ".xlsx"]:

        loader = UnstructuredFileLoader(save_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})
        # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever,
        #                                  return_source_documents=True)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever)
        return jsonify({'answer': '上传成功'})
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)})


@app.route('/api/answer', methods=['POST'])
def answer():
    global chat_history
    try:
        question = request.json['message']
        print(question)
        answer = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, answer["answer"]))
        # answer = qa({"query": question})['result']
        print(chat_history)
        print(answer)
        return jsonify({'answer': answer["answer"]})
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)})

@app.route("/")
def root():
    return render_template("index_pdf.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
