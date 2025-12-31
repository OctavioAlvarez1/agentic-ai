from workflow_agents.base_agents import RAGKnowledgePromptAgent
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# 🔹 Minimal knowledge to avoid heavy computation
knowledge_text = "Clara hosts a podcast called Crosscurrents about science, culture, and ethics."

persona = "You are a knowledge-based assistant."

rag_agent = RAGKnowledgePromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    chunk_size=50,     # MUY chico
    chunk_overlap=0
)

# Run minimal RAG pipeline
rag_agent.chunk_text(knowledge_text)
rag_agent.calculate_embeddings()

prompt = "What is the podcast that Clara hosts about?"
response = rag_agent.find_prompt_in_knowledge(prompt)

print("Prompt:", prompt)
print("Response:", response)
print("\nExplanation: The agent retrieved the answer using embeddings and similarity search over the provided knowledge.")
