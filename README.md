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

## Prompt

[How to Prompt Llama 2](https://huggingface.co/blog/llama2#how-to-prompt-llama-2)에서 설명하고 있는 Prompt는 아래와 같습니다.

```text
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
```

이를 사용할때는 아래와 같습니다.

```text
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden 😱 What should I do? [/INST]
```

또한 call history를 고려하면 아래와 같습니다.

```text
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
```

## Conversation

대화(Conversation)에서 chat history를 이용하기 위해서는 아래 방안1의 [ConversationBufferMemory](https://python.langchain.com/docs/modules/memory/types/buffer)을 이용하는 방법과 방안 2와 같이 chat history의 length를 직접 관리하면서 history를 PromptTemplate을 이용하여 prompt에 포함하는 방법이 있습니다. [lambda-chat](./lambda-chat/lambda_function.py)의 methodOfConversation을 이용하여 ConversationChain 또는 PromptTemplate을 선택할 수 있습니다.

### 방안1: ConversationChain을 이용

[ConversationBufferMemory](https://python.langchain.com/docs/modules/memory/types/buffer)을 이용하여 대화 이력(chat history)를 저장하고, [ConversationChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversation.base.ConversationChain.html)
을 이용하여 history를 관리합니다.

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, verbose=True, memory=memory
)
```

아후 아래처럼 input인 text에 대해 대화(conversation)을 chat history를 포함하여 구현할 수 있습니다.

```python
msg = conversation.predict(input=text)
```

### 방안2: PromptTemplate을 이용하는 방법

[ConversationBufferMemory](https://python.langchain.com/docs/modules/memory/types/buffer)을 이용하여 대화 이력(chat history)를 저장합니다.

```python
from langchain.memory import ConversationBufferMemory
chat_memory = ConversationBufferMemory(human_prefix='Human', ai_prefix='AI')
```

memory에서 chat history를 분리합니다. 이때, prompt의 context 크기를 고려하여 history를 chunk로 나누어서 가장 최신의 2개 chunk를 prompt로 활용합니다.

```python
def get_answer_using_chat_history(query, chat_memory):  
    condense_template = """Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {chat_history}
    
    Human: {question}
    AI:"""
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
```

아래처럼 결과는 chat_memory에 저장하여 대화(conversation)에서 활용합니다.

```python
msg = get_answer_using_chat_history(text, chat_memory)
chat_memory.save_context({"input": text}, {"output": msg})
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

"I live in Seoul, Korea." 라고 입력합니다.

![image](https://github.com/kyopark2014/Llama2-conversational-chatbot/assets/52392004/f489d8b7-7d9d-4fc0-a0ad-2ffd9671232a)

이후 "Tell me how to travel the city."와 같이 대명사로 물었을때에 이전 chat history로 아래처럼 응답합니다.

![image](https://github.com/kyopark2014/Llama2-conversational-chatbot/assets/52392004/654d4fa5-876f-4f59-8d69-81b9ce05b510)



### 리소스 정리하기

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. [Cloud9 console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래와 같이 삭제를 합니다.

```java
cdk destroy
```


## 결론

SageMaker JumpStart를 이용하여 대규모 언어 모델(LLM)인 LLama 2를 쉽게 배포하였고, DynamoDB를 이용하여 이전 대화 이력을 조회하여 대화가 가능한 Chatbot을 구현하였습니다. Amazon SageMaker JumpStart는 다양한 종류의 언어 모델을 가지고 있으므로 목적에 맞게 선택하여 사용할 수 있습니다. 여기서는 Llama 2을 이용하여 chatbot을 구현하였고, 또한 Chatbot 어플리케이션 개발을 위해 LangChain을 활용하였고, IaC(Infrastructure as Code)로 AWS CDK를 이용하였습니다. 대용량 언어 모델은 향후 다양한 어플리케이션에서 효과적으로 활용될것으로 기대됩니다. SageMaker JumpStart을 이용하여 대용량 언어 모델을 개발하면 기존 AWS 인프라와 손쉽게 연동하고 다양한 어플리케이션을 효과적으로 개발할 수 있습니다.



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
