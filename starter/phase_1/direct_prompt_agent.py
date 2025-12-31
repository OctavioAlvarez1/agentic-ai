# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

# Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)

# Use direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# Print an explanatory message describing the knowledge source used by the agent
print(
    "\nExplanation: This agent uses general world knowledge learned by the "
    "underlying language model (gpt-3.5-turbo) during training. "
    "No additional persona or external knowledge was provided."
)
