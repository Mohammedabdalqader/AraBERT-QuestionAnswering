[//]: # (Image References)

[postman-ask]: postman/postman-ask.png "ask"
[postman-upload]: postman/postman-upload.png "upload"


# Pretraining/Finetuning AraElectra :blush:

In this repository, I show how to pre-train the AraElectra model on a specific domain and then how to use it for downstream tasks, such as question answering system. There are 2 notebooks:

1- Pretrain_AraElectra.ipynb

used to pretrain the AraElectra model on a user-defined dataset. In this notebook, the dataset is first prepared for training in the expected format and then the pretraining can be started. At the end of the training process, you can convert the Tensorflow checkpoint to a Pytorch model and store it in a suitable
location namly: "araelectra-base-discriminator/" to be used later for fine-tuning this model on a question answering system. 

The expexted dataset structure is a text-file in which each line represent a single sentence ends with '.' and between the paragraphs there is an empty line. An example for the dataset will be found in dataset/pretraining-dataset/iskan.txt.

2- Finetune-AraElectra-QA.ipynb

Through this notebook you can fine-tune the AraElectra model for a question-answer system. This notebook contains all the necessary processes from preparing the dataset to performing the inference. The dataset must be in this format:

```
QA_data.json
├── "data"
│   └── [i]
│       ├── "paragraphs"
│       │   └── [j]
│       │       ├── "context": "paragraph text"
│       │       └── "qas"
│       │           └── [k]
│       │               ├── "answers"
│       │               │   └── [l]
│       │               │       ├── "answer_start": N
│       │               │       └── "text": "answer"
│       │               ├── "id": "<uuid>"
│       │               └── "question": "paragraph question?"
│       └── "title": "document id"
└── "version": 1.1

```

As an annotation tool for the Question Answering System I recommend Haystack from here: https://annotate.deepset.ai/

## Getting Started

### Clone das Repo

	git clone https://github.com/Mohammedabdalqader/AraElectra-QuestionAnswering.git
	cd AraElectra-QuestionAnswering


### Create Conda Environment (Recommended)
    conda create -n [env name] python=3.7
    conda activate [env name]
    pip install -r requirements.txt

### Install python Requirements without Conda Environment (Not Recommended)

	pip3 install -requirements.txt
    

### Install pre-commit Hooks

    pre-commit install



### Install the Git LFS client:

- For Linux and Mac OS X, use a package manager to install git-lfs, or download from (https://git-lfs.github.com/). 

- On Windows, download the installer from here, and run it (https://git-lfs.github.com/).

Now you need to run this line:

    git lfs install

### Download pretrained AraElectra Model which trained on SQuAD & arcd datasets

You can download the model from here: https://drive.google.com/file/d/1-l2-rNnui7UulEEdUdG-POSbKyNb6DHk/view?usp=sharing
The model must be saved in araelectra-QA-model/

### Run the Flask app

    gunicorn -k gevent -b 0.0.0.0:8080 app.api.routes:app -t 300

### Use Postman to ask questions and upload new dokuments

- asking question endpoint: http://0.0.0.0:8080/ask
  ![ask][postman-ask]
- upload new documents (.txt) endpoint: http://0.0.0.0:8080/upload_new_document
  ![upload][postman-upload]


### References

 - AraBERT: https://github.com/aub-mind/arabert                                                                 
 - BERT: https://github.com/google-research/bert

