from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


prompt1 = PromptTemplate(
    template = 'Generate a detail report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template='Generate a detailed summary from the following test \n{text}',
    input_variables=['topic']

)

model = ChatOpenAI()

parser = StrOutputParser()

# Chain prompt1 -> model -> prompt2 -> model -> parser

chain = prompt1 | model | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in the UK'})

print(result)