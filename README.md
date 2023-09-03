# Llama 2로 Conversational Chatbot 만들기

여기서는 [Llama 2의 대규모 언어 모델(Large Language Models)](https://aws.amazon.com/ko/blogs/machine-learning/llama-2-foundation-models-from-meta-are-now-available-in-amazon-sagemaker-jumpstart/)을 이용하여 대화가 가능한 chatbot을 [vector store](https://python.langchain.com/docs/modules/data_connection/vectorstores/)를 구현합니다. 대량의 데이터로 사전학습(pretrained)한 대규모 언어 모델(LLM)은 학습되지 않은 질문에 대해서도 가장 가까운 답변을 맥락(context)에 맞게 찾아 답변할 수 있습니다. 하지만 대화(conversation)을 위해서는 기존 대화를 Prompt로 활용할 수 있어야 하므로, LangChain의 chain과 prompt template를 이용합니다. 


## 아키텍처 개요


<img src="https://github.com/kyopark2014/Llama2-conversational-chatbot/assets/52392004/86c8ada9-a72e-488f-a941-3a5710edb688" width="800">


## 주요 구성

### LangChain 이용하기

LangChain을 이용해서 Llama 2에 연결하는 경우에 아래와 같이 endpoint_kwargs에 CustomAttributes를 추가합니다. 

```python
endpoint_name = os.environ.get('endpoint')

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
    "max_new_tokens": 256, 
    "top_p": 0.9, 
    "temperature": 0.6
} 

llm = SagemakerEndpoint(
    endpoint_name = endpoint_name, 
    region_name = aws_region, 
    model_kwargs = parameters,
    endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    content_handler = content_handler
)
```

### Embedding

[SageMaker Endpoint Embeddings](https://python.langchain.com/docs/integrations/text_embedding/sagemaker-endpoint)에 따라 아래와 같이 embedding을 정의합니다. 여기서 embedding용 SageMaker Endpoint인 endpoint_embedding는 SageMaker JumpStart에서 "GPT-J 6B Embedding FP16"을 설치한 후에 생성된 Endpoint입니다.

```python
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from typing import Dict, List
class ContentHandler2(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({ "text_inputs": inputs, ** model_kwargs})
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
```

#### Conversation

대화(Conversation)을 위해서는 Chat History를 이용한 Prompt Engineering이 필요합니다. 여기서는 Chat History를 위한 chat_memory와 RAG에서 document를 retrieval을 하기 위한 memory를 이용합니다.

```python
# memory for conversation
chat_memory = ConversationBufferMemory(human_prefix='Human', ai_prefix='AI')

# memory for retrival docs
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key='answer', human_prefix='Human', ai_prefix='AI')
```

Chat history를 위한 condense_template과 document retrieval시에 사용하는 prompt_template을 아래와 같이 정의하고, [ConversationalRetrievalChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html)을 이용하여 아래와 같이 구현합니다.


```python
def get_answer_using_template_with_history(query, vectorstore, chat_memory):  
    condense_template = """Given the following conversation and a follow up question, answer friendly. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Chat History:
    {chat_history}
    Human: {question}
    AI:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),         
        condense_question_prompt=CONDENSE_QUESTION_PROMPT, # chat history and new question
        chain_type='stuff', # 'refine'
        verbose=False, # for logging to stdout
        rephrase_question=True,  # to pass the new generated question to the combine_docs_chain
        
        memory=memory,
        #max_tokens_limit=300,
        return_source_documents=True, # retrieved source
        return_generated_question=False, # generated question
    )

    # combine any retrieved documents.
    prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    AI:"""
    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template) 
    
    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history = chats['history']
    print('chat_history: ', chat_history)

    # make a question using chat history
    result = qa({"question": query, "chat_history": chat_history})    
    print('result: ', result)    
    
    # get the reference
    source_documents = result['source_documents']
    print('source_documents: ', source_documents)

    if len(source_documents)>=1 and enableReference == 'true':
        reference = get_reference(source_documents)
        #print('reference: ', reference)
        return result['answer']+reference
    else:
        return result['answer']
```        


### AWS CDK로 인프라 구현하기

[CDK 구현 코드](./cdk-chatbot-llama2/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)


### CDK를 이용한 인프라 설치
[인프라 설치](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 





### 실행결과




### 리소스 정리하기

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. [Cloud9 console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래와 같이 삭제를 합니다.

```java
cdk destroy
```


## 결론

SageMaker JumpStart를 이용하여 대규모 언어 모델(LLM)인 LLama 2를 쉽게 배포하였고, vector store를 이용하여 질문과 답변(Question/Answering)을 수행하는 chatbot의 기능을 향상시켰습니다. Amazon SageMaker JumpStart는 다양한 종류의 언어 모델을 가지고 있으므로 목적에 맞게 선택하여 사용할 수 있습니다. 여기서는 Llama 2을 이용하여 RAG 동작을 구현하였고, 대용량 언어 모델의 환각(hallucination) 문제를 해결할 수 있었습니다. 또한 Chatbot 어플리케이션 개발을 위해 LangChain을 활용하였고, IaC(Infrastructure as Code)로 AWS CDK를 이용하였습니다. 대용량 언어 모델은 향후 다양한 어플리케이션에서 효과적으로 활용될것으로 기대됩니다. SageMaker JumpStart을 이용하여 대용량 언어 모델을 개발하면 기존 AWS 인프라와 손쉽게 연동하고 다양한 어플리케이션을 효과적으로 개발할 수 있습니다.



## Reference 

[Fundamentals of combining LangChain and Amazon SageMaker (with Llama 2 Example)](https://medium.com/@ryanlempka/fundamentals-of-combining-langchain-and-sagemaker-with-a-llama-2-example-694924ab0d92)

## Referecne: SageMaker Endpoint로 구현하기

SageMaker Endpoint를 직접 호출하여 prompt 응답을 받는 함수입니다.

```python
def get_llm(text):
    dialog = [{"role": "user", "content": text}]

    parameters = {
        "max_new_tokens": 256, 
        "top_p": 0.9, 
        "temperature": 0.6
    } 

    payload = {
        "inputs": [dialog], 
        "parameters":parameters
    }
    
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType='application/json', 
        Body=json.dumps(payload).encode('utf-8'),
        CustomAttributes="accept_eula=true",
    )                

    body = response["Body"].read().decode("utf8")
    body_resp = json.loads(body)
    print(body_resp[0]['generation']['content'])

    return body_resp[0]['generation']['content']
```
