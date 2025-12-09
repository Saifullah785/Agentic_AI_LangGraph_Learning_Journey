from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
import requests
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv


load_dotenv()

search_tool = DuckDuckGoSearchResults()

@Tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city using a public API.
    """
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    
    response = requests.get(url)
    
    return response.json()


llm = ChatOpenAI()

# step 2: Pull the React prompt from LangChain Hub

prompt = hub.pull("hwchase17/react") # pull the standard ReAct prompt


# step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)
# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
    max_iterations=5
)



# what is the release date of Dhadak 2?
# what is the current temp of gurgaon
# Identify the birthplace city of kalpana chawla (search) and give its current temperature.

# Step 5: Invoke 
response = agent_executor.invoke({"input": "What is the current temp of gurgaon"})
print(response)

print(response['output'])