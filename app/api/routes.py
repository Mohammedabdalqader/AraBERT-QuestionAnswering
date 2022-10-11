# import packages
import json
import os
import logging
from flask_cors import CORS
from flask import Flask, request, jsonify
from haystack.utils import launch_es
from haystack.utils import clean_wiki_text, convert_files_to_docs
import os
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, ElasticsearchRetriever
from haystack.nodes import TransformersReader
from haystack.pipelines import ExtractiveQAPipeline

from src.finetuningQA.preprocess import ArabertPreprocessor



#application settings
app = Flask(__name__)
CORS(app)

# Application directory for inputs and training
app.config["input_document"] = "documents/alshura"
app.config["qa_model"] = "araelectra-QA-model/"

# ElasticSearch server host information
app.config["host"] = "0.0.0.0"
app.config["username"] = ""
app.config["password"] = ""
app.config["port"] = "9200"

model_name = "aubmindlab/araelectra-base-discriminator"
arabert_prep = ArabertPreprocessor(model_name=model_name)

def initialize_reader():
    # using pretrain model
    reader = TransformersReader(model_name_or_path="araelectra-QA-model/", tokenizer="araelectra-QA-model/", use_gpu=False)
    return reader

def initialize_haystack_elasticsearch_document_storage():
    launch_es()
    # Get the host where Elasticsearch is running, default to localhost
    document_store = ElasticsearchDocumentStore(
        host=app.config["host"],
        username=app.config["username"],
        password=app.config["password"],
        index="document")
    document_store.delete_documents()
    return document_store

def prepare_document():
    document_store = initialize_haystack_elasticsearch_document_storage()
    # Convert files to dicts
    # You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers). For more details see convert_files_to_docs function.
    # It must take a str as input, and return a str.
    docs = convert_files_to_docs(dir_path=app.config["input_document"], split_paragraphs=True)

    # We now have a list of dictionaries that we can write to our document store.
    # If your texts come from a different source (e.g. a DB), you can of course skip convert_files_to_dicts() and create the dictionaries yourself.
    # The default format here is:
    # {
    #    'content': "<DOCUMENT_TEXT_HERE>",
    #    'meta': {'name': "<DOCUMENT_NAME_HERE>", ...}
    # }
    # (Optionally: you can also add more key-value-pairs here, that will be indexed as fields in Elasticsearch and
    # can be accessed later for filtering or shown in the responses of the Pipeline)

    # Now, let's write the dicts containing documents to our DB.
    document_store.write_documents(docs)
    return document_store


document_store = prepare_document()
print("Documents are ready ...")
reader = initialize_reader()
print("QA model has been initialized ...")

@app.route('/upload_new_document', methods=['POST'])
def upload_new_document():
    global document_store
    """Add a new document to the Elastic search Document Store."""
    if request.files:

        # uploaded document for target source
        doc = request.files["document"]

        file_path = os.path.join(app.config["input_document"], doc.filename)

        # saving the file to the input directory
        doc.save(file_path)

        #initialization of the Haystack Elasticsearch document storage
        document_store = prepare_document()

        return json.dumps(
            {'status':'Susccess','message':
                'document available at http://'+ app.config["host"] +':'
                + app.config["port"],
                'result': []})
    else:
        return json.dumps({'status':'Failed','message': 'No file uploaded', 'result': []})

@app.route('/ask', methods=['POST'])
def ask():
    ''''

    {"question":"من هم الفئات المستهدفة للمكتبة البرلمانية"}
    {"question":"كم مضى على إنطلاق السلطة التشريعية"}
    {"question":"في أي عام أنشئت المكتبة البرلمانية"}
    {"question":"بماذا تعرف المكتبة البرلمانية"}
    {"question":"ما هي رسالة المكتبة البرلمانية"}
    {"question":"بماذا تتمثل رؤية المكتبة البرلمانية"}
    {"question":"ما هي أوقات عمل المكتبة"}
    {"question":"أذكر تعليمات المكتبة"}

    {"question":"ما هو الإسم الذي يطلق على مجلس الشيوخ"}
    {"question":"من هي أول وزيرة في مجلس الوزراء"}
    {"question":"في أي عام اصبحت ندى الحافظ وزيرة للصحة"}
    {"question":" من هو وزير التجارة "}
    {"question":"من هي سميرة رجب"}
    {"question":"من هو رئيس كتلة المنبر الإسلامي"}
    {"question":"ما هي الفترة التي كان فيها فيصل فولاذ وزيرا لشؤن حقوق الإنسان"}

    {"question":"ما هي قيم الأمانة العامة لمجلس الشورى"}
    {"question":"بماذا تتمثل رؤية الأمانة العامة لمجلس الشورى"}

    {"question":"متى عقد الاجتماع التشاوري لمجالس الدول الأعضاء في منظمة التعاون الإسلامي"}
    {"question":"أين عقد الاجتماع التشاوري لمجالس الدول الأعضاء في منظمة التعاون الإسلامي"}
    {"question":"من الذي مثّل الشعبة البرلمانية"}
    {"question":"ما هي المبادئ التي سجلت في ميثاق للعمل الوطني"}
    {"question":"في أي عام أصدرت البحرين دستورها"}
    {"question":"متى تم حل المجلس الوطني ونقل السلطة التشريعية إلى صاحب السمو الأمير ومجلس الوزراء "}
    {"question":"ما هي مدة المجلس الوطني "}
    {"question":"خلال كم شهر تم اجراء الانتخابات للمجلس الجديد "}
    {"question":"ما هي العبارة التي اضافتها المادة 63 "}
    {"question":"على ماذا تنص المادة 62 "}

    '''

    """Return the n answers."""
    print("recieved a question")
    question = request.json['question']
    question = arabert_prep.preprocess(question)

    #initialization of BM25Retriever
    retriever = BM25Retriever(document_store= document_store)
    # ExtractiveQAPipeline sticks together reader and retriever
    # in a pipeline to answer our actual questions.
    pipe = ExtractiveQAPipeline(reader, retriever)


    prediction = pipe.run(
    query=question, params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}}
)
    answer = []
    for res in prediction['answers']:
        answer.append(res.answer)
    return json.dumps({'status':'success','message': 'Process succesfully', 'result': answer[0]}, ensure_ascii=False)