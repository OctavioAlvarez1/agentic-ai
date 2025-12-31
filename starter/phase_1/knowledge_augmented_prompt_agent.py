# Import the KnowledgeAugmentedPromptAgent class from workflow_agents
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris."

# Instantiate a KnowledgeAugmentedPromptAgent
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# Send the prompt to the agent
response = knowledge_agent.respond(prompt)

# Print the response
print(response)

# Print a statement demonstrating the use of provided knowledge
print(
    "\nExplanation: The agent used only the explicitly provided knowledge "
    "to answer the question, even though it contradicts real-world facts, "
    "showing that it ignored its inherent LLM knowledge."
)
