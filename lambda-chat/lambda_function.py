import json
import boto3
import os
import time
import datetime
from io import BytesIO
import PyPDF2
import csv
import sys
import re

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain.vectorstores import FAISS
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import CSVLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
endpoint_llm = os.environ.get('endpoint_llm')
endpoint_embedding = os.environ.get('endpoint_embedding')

enableConversationMode = os.environ.get('enableConversationMode', 'enabled')
print('enableConversationMode: ', enableConversationMode)

methodOfConversation = 'PromptTemplate' # ConversationChain or PromptTemplate

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({
            "inputs" : 
            [
                [
                    {
                        "role" : "system",
                        "content" : "You are a kind robot."
                    },
                    {
                        "role" : "user", 
                        "content" : prompt
                    }
                ]
            ],
            "parameters" : {**model_kwargs}})
        return input_str.encode('utf-8')
      
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

content_handler = ContentHandler()
aws_region = boto3.Session().region_name
client = boto3.client("sagemaker-runtime")
parameters = {
    "max_new_tokens": 1024, 
    "top_p": 0.9, 
    "temperature": 0.1
} 
HUMAN_PROMPT = "\n\nUser:"
AI_PROMPT = "\n\nAssistant:"

llm = SagemakerEndpoint(
    endpoint_name = endpoint_llm, 
    region_name = aws_region, 
    model_kwargs = parameters,
    endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    content_handler = content_handler
)

map = dict() # Conversation

# Conversation
if methodOfConversation == 'ConversationChain':
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm, verbose=True, memory=memory
    )
elif methodOfConversation == 'PromptTemplate':
    # memory for conversation
    chat_memory = ConversationBufferMemory(human_prefix='User', ai_prefix='Assistant')

# Embedding
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from typing import Dict, List
class ContentHandler2(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["embedding"]

content_handler2 = ContentHandler2()
embeddings = SagemakerEndpointEmbeddings(
    endpoint_name = endpoint_embedding,
    region_name = aws_region,
    content_handler = content_handler2,
)

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(texts):    
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+') 
    word_kor = pattern_hangul.search(str(texts))
    print('word_kor: ', word_kor)
    
    if word_kor:
        #prompt_template = """\n\nHuman: 다음 텍스트를 간결하게 요약하세오. 텍스트의 요점을 다루는 글머리 기호로 응답을 반환합니다.
        prompt_template = """\n\nUser: 다음 텍스트를 요약해서 500자 이내로 설명하세오.

        {text}
        
        Assistant:"""        
    else:         
        prompt_template = """\n\nUser: Write a concise summary of the following:

        {text}
        
        Assistant:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    docs = [
        Document(
            page_content=t
        ) for t in texts[:3]
    ]
    summary = chain.run(docs)
    print('summary: ', summary)

    if summary == '':  # error notification
        summary = 'Fail to summarize the document. Try agan...'
        return summary
    else:
        # return summary[1:len(summary)-1]   
        return summary

def get_answer_using_chat_history(query, chat_memory):  
    condense_template = """<s>[INST] <<SYS>>
    Using the following conversation between the Assistant and User, answer friendly for the newest question. 

    {chat_history}
    
    If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor. <</SYS>>

    {question} [/INST]"""
    #Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor.

    #You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please #ensure that your responses are socially unbiased and positive in nature.    
    #If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<</SYS>>
    
    #Here are some previous conversations between the Assistant and User:
    #Here is the latest conversation between Assistant and User.
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
        
    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history_all = chats['history']
    print('chat_history_all: ', chat_history_all)

    # use last two chunks of chat history
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
    texts = text_splitter.split_text(chat_history_all) 

    pages = len(texts)
    print('pages: ', pages)

    if pages >= 2:
        chat_history = f"{texts[pages-2]} {texts[pages-1]}"
    elif pages == 1:
        chat_history = texts[0]
    else:  # 0 page
        chat_history = ""
    print('chat_history:\n ', chat_history)

    # make a question using chat history
    if pages >= 1:
        result = llm(CONDENSE_QUESTION_PROMPT.format(question=query, chat_history=chat_history))
    else:
        result = llm(query)
    print('result: ', result)

    return result    


system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

Llama2_BASIC_PROMPT = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{question} [/INST]"""

Llama2_HISTORY_PROMPT = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{history}
<s>[INST] {question} [/INST]"""

message_with_history = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

I live in Seoul. [/INST] 
Great! I'm glad to hear that you live in Seoul! Seoul is a fascinating city with a rich history and culture. Is there anything specific you would like to know or discuss about Seoul or Korea in general? I'm here to help and provide information that is safe, respectful, and socially unbiased. Please feel free to ask! <s>
<s>[INST] Tell me how to travel the places. [/INST]"""

def lambda_handler(event, context):
    print(event)
    userId  = event['user-id']
    print('userId: ', userId)
    requestId  = event['request-id']
    print('requestId: ', requestId)
    type  = event['type']
    print('type: ', type)
    body = event['body']
    print('body: ', body)

    global llm, chat_memory, map
    global enableConversationMode  # debug

    if userId in map:
        chat_memory = map[userId]
        print('chat_memory exist. reuse it!')
    else: 
        chat_memory = ConversationBufferMemory(human_prefix='User', ai_prefix='Assistant')
        map[userId] = chat_memory
        print('chat_memory does not exist. create new one!')

    if methodOfConversation == 'ConversationChain':
        conversation = ConversationChain(llm=llm, verbose=True, memory=chat_memory)
       
    start = int(time.time())    

    msg = ""
    if type == 'text':
        text = body

        # debugging
        if text == 'enableConversationMode':
            enableConversationMode = 'true'
            msg  = "Conversation mode is enabled"
        elif text == 'disableConversationMode':
            enableConversationMode = 'false'
            msg  = "Conversation mode is disabled"
        else:

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")
                
            if enableConversationMode == 'true':
                if methodOfConversation == 'ConversationChain':
                    msg = conversation.predict(input=text)
                elif methodOfConversation == 'PromptTemplate':
                    msg = get_answer_using_chat_history(text, chat_memory)

                    storedMsg = str(msg).replace("\n"," ") 
                    chat_memory.save_context({"input": text}, {"output": storedMsg})         
            else:
                #msg = llm(HUMAN_PROMPT+text+AI_PROMPT)
                #msg = llm(Llama2_BASIC_PROMPT.format(system_prompt=system_prompt, question=text))
                msg = llm(message_with_history)
            
    elif type == 'document':
        object = body
        
        file_type = object[object.rfind('.')+1:len(object)]
        print('file_type: ', file_type)
            
        if file_type == 'csv':
            docs = load_csv_document(object)
            texts = []
            for doc in docs:
                texts.append(doc.page_content)
            print('texts: ', texts)
        else:
            texts = load_document(file_type, object)
            
        msg = get_summary(texts)            
                
    elapsed_time = int(time.time()) - start
    print("total run time(sec): ", elapsed_time)

    print('msg: ', msg)

    item = {
        'user-id': {'S':userId},
        'request-id': {'S':requestId},
        'type': {'S':type},
        'body': {'S':body},
        'msg': {'S':msg}
    }

    client = boto3.client('dynamodb')
    try:
        resp =  client.put_item(TableName=callLogTableName, Item=item)
    except: 
        raise Exception ("Not able to write into dynamodb")
        
    print('resp, ', resp)

    return {
        'statusCode': 200,
        'msg': msg,
    }
