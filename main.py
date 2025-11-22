# pip install -qU "langchain[anthropic]" to call the model

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=ChatOpenAI(model_name="gpt-5-nano",
    api_key=os.environ.get("OPENAI_API_KEY")),
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
invoked = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

for message in invoked["messages"]:
    print(f"\n[{message.type.upper()}]: {message.content}")

# Messed with Slack integration but not ready yet to build it with MVP. Going to do something more simpler first