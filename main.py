from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

os.environ["AWS_PROFILE"] = "thornjm@amazon.com"

#bedrock client

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

modelID = "amazon.titan-text-express-v1"

llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
)

def eero_chabot_two(language,freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. Your are in {language}.\n\n{freeform_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response=bedrock_chain({'language':language, 'freeform_text':freeform_text})
    return response

#print(eero_chabot_two("english", "how do I set up my eero router?"))

st.title("eero CS Chatbot")

language = st.sidebar.selectbox("language", ["english", "spanish", "french", "japanese"])

if language:
    freeform_text = st.sidebar.text_area(label="What is your question?",max_chars=150)

if freeform_text:
    response = eero_chabot_two(language,freeform_text)
    st.write(response['text'])

