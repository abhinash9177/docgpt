import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain. text_splitter import CharacterTextSplitter
from langchain. vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api', methods=['POST'])
def get_data():
    data = request.get_json()
    question = data.get('question')
    folder_path = data.get('userId')
    #folder_path = 'abhi'
    file_name = data.get('fileName')
    file_path = os.path.join(folder_path, file_name)
    os.environ["OPENAI_API_KEY"] = "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"
    # Fetch the PDF file from the URL
    url = "https://www.imf.org/external/pubs/ft/ar/2022/downloads/2022-financial-statements.pdf"


    #response = requests.get(url)
    #pdf_file = BytesIO(response.content)
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        num_pages = len(reader.pages)
        print(f"The PDF file has {num_pages} pages.")

        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000, 
            chunk_overlap = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings()

        docsearch = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        query = "give me the summary"
        docs = docsearch.similarity_search(question)
        answer = chain.run(input_documents=docs, question=query)
        data = {
            'answer': answer,
            'age': 30,
            'city': 'your question is: '+ question
        }
        return jsonify(data)


#upload file

@app.route('/upload', methods=['POST'])
def upload_file():
    folder_name = request.form['folder_name']

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
       
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Store file in uploads folder
    filename = file.filename
    file.save(os.path.join(folder_name, filename))

    return jsonify({'message': 'File uploaded successfully'}), 200

#get files in folder
@app.route('/getfilenames', methods=['GET'])
def get_file_names():
    folder_path = request.args.get('folder')
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return jsonify({'file_names': file_names})

if __name__ == '__main__':
    app.run()