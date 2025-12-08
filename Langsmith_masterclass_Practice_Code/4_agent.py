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
