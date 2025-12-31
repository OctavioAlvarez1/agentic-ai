from openai import OpenAI
import os
import numpy as np
import re
import csv
import uuid
import pandas as pd
from datetime import datetime

# -------------------------
# DirectPromptAgent
# -------------------------
class DirectPromptAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content


# -------------------------
# AugmentedPromptAgent
# -------------------------
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"{self.persona} Forget all previous context."
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# -------------------------
# KnowledgeAugmentedPromptAgent
# -------------------------
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, input_text):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        system_prompt = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge:\n"
            f"{self.knowledge}\n"
            "Answer the prompt based on this knowledge, not your own."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# -------------------------
# EvaluationAgent
# -------------------------
class EvaluationAgent:
    def __init__(self, openai_api_key, persona, evaluation_criteria, agent_to_evaluate, max_interactions=5):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        prompt_to_evaluate = initial_prompt
        final_response = None
        evaluation_result = None

        for i in range(self.max_interactions):
            response_from_worker = self.agent_to_evaluate.respond(prompt_to_evaluate)
            final_response = response_from_worker

            eval_prompt = (
                f"Does the following answer:\n{response_from_worker}\n\n"
                f"Meet this criteria:\n{self.evaluation_criteria}\n\n"
                "Respond Yes or No, and explain why."
            )

            eval_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )

            evaluation_result = eval_response.choices[0].message.content.strip()

            if evaluation_result.lower().startswith("yes"):
                break

            instruction_prompt = f"Provide instructions to fix the answer based on these reasons:\n{evaluation_result}"

            instruction_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": instruction_prompt}
                ],
                temperature=0
            )

            instructions = instruction_response.choices[0].message.content.strip()

            prompt_to_evaluate = (
                f"The original prompt was:\n{initial_prompt}\n\n"
                f"The previous response was:\n{response_from_worker}\n\n"
                f"Apply only the following corrections:\n{instructions}"
            )

        return {
            "final_response": final_response,
            "evaluation": evaluation_result,
            "iterations": i + 1
        }


# -------------------------
# RoutingAgent
# -------------------------
class RoutingAgent:
    def __init__(self, openai_api_key, agents):
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def route(self, user_input):
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "No suitable agent found."

        return best_agent["func"](user_input)


# -------------------------
# ActionPlanningAgent
# -------------------------
class ActionPlanningAgent:
    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract from "
            "the user prompt the steps requested to complete the action. "
            "Only return the steps. Forget any previous context. "
            f"This is your knowledge:\n{self.knowledge}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        steps = [s.strip() for s in response.choices[0].message.content.split("\n") if s.strip()]
        return steps


# -------------------------
# RAGKnowledgePromptAgent
# -------------------------
class RAGKnowledgePromptAgent:
    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start
            })
            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id", "text", "chunk_size"])
            writer.writeheader()
            for c in chunks:
                writer.writerow(c)

        return chunks

    def calculate_embeddings(self):
        df = pd.read_csv(f"chunks-{self.unique_filename}")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}")
        df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(prompt_embedding, emb)
        )

        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."
                },
                {
                    "role": "user",
                    "content": f"Answer based only on this information:\n{best_chunk}\n\nPrompt: {prompt}"
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content
