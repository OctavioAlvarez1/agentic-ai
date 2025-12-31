# Phase 2 - Agentic Workflow

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent
)

import os
from dotenv import load_dotenv

# -----------------------------
# Environment setup
# -----------------------------

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# -----------------------------
# Load Product Specification
# -----------------------------

with open("phase_2/Product-Spec-Email-Router.txt", "r", encoding="utf-8") as f:
    product_spec = f.read()

# -----------------------------
# TODO 4 - Action Planning Agent
# -----------------------------

knowledge_action_planning = (
    "You are a Technical Program Manager. "
    "You break down a high-level product request into clear, sequential steps "
    "that can be executed by Product Managers, Program Managers, and Development Engineers."
)

action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning
)

# -----------------------------
# TODO 5 - Product Manager Knowledge
# -----------------------------

persona_product_manager = (
    "You are a Product Manager responsible for defining user personas and user stories."
)

knowledge_product_manager = (
    "You define user stories strictly based on the following product specification. "
    "Do not define features or technical tasks.\n\n"
    + product_spec
)

# -----------------------------
# TODO 6 - Product Manager Knowledge Agent
# -----------------------------

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)

# -----------------------------
# TODO 7 - Product Manager Evaluation Agent
# -----------------------------

persona_product_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)

evaluation_criteria_product_manager = (
    "The answer must ONLY contain user stories written EXACTLY in this format:\n"
    "As a [type of user], I want [an action or feature] so that [benefit/value].\n"
    "Do not include explanations, summaries, or analysis."
)

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    agent_to_evaluate=product_manager_knowledge_agent,
    max_interactions=5
)

# -----------------------------
# Program Manager Knowledge Agent
# -----------------------------

persona_program_manager = (
    "You are a Program Manager responsible for defining product features "
    "based on validated user stories and the product specification."
)

knowledge_program_manager = (
    "You define product features based on user stories and the product specification. "
    "Do not define user stories or technical tasks.\n\n"
    + product_spec
)

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)

# -----------------------------
# TODO 8 - Program Manager Evaluation Agent
# -----------------------------

persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)

evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    agent_to_evaluate=program_manager_knowledge_agent,
    max_interactions=5
)

# -----------------------------
# Development Engineer Knowledge Agent
# -----------------------------

persona_dev_engineer = (
    "You are a Development Engineer responsible for defining detailed technical tasks "
    "required to implement product features."
)

knowledge_dev_engineer = (
    "You define engineering tasks based on product features and the product specification. "
    "Do not define user stories or product features.\n\n"
    + product_spec
)

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

# -----------------------------
# TODO 9 - Development Engineer Evaluation Agent
# -----------------------------

persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)

evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure:\n"
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    agent_to_evaluate=development_engineer_knowledge_agent,
    max_interactions=5
)

# -----------------------------
# TODO 11 - Support Functions
# -----------------------------

def product_manager_support_function(step):
    response = product_manager_knowledge_agent.respond(step)
    evaluation = product_manager_evaluation_agent.evaluate(response)
    return evaluation["final_response"]


def program_manager_support_function(step):
    response = program_manager_knowledge_agent.respond(step)
    evaluation = program_manager_evaluation_agent.evaluate(response)
    return evaluation["final_response"]


def development_engineer_support_function(step):
    response = development_engineer_knowledge_agent.respond(step)
    evaluation = development_engineer_evaluation_agent.evaluate(response)
    return evaluation["final_response"]

# -----------------------------
# TODO 10 - Routing Agent
# -----------------------------

routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=[
        {
            "name": "Product Manager",
            "description": (
                "Responsible for defining product personas and user stories only. "
                "Does not define features or technical tasks."
            ),
            "func": lambda step: product_manager_support_function(step)
        },
        {
            "name": "Program Manager",
            "description": (
                "Responsible for defining product features based on user stories. "
                "Does not define user stories or technical tasks."
            ),
            "func": lambda step: program_manager_support_function(step)
        },
        {
            "name": "Development Engineer",
            "description": (
                "Responsible for defining detailed engineering tasks required "
                "to implement product features."
            ),
            "func": lambda step: development_engineer_support_function(step)
        }
    ]
)

# -----------------------------
# TODO 12 - Execute Agentic Workflow
# -----------------------------

workflow_prompt = """
Create a complete project plan for the Email Router product.
First define user personas and user stories.
Then define product features.
Finally define detailed engineering tasks.
"""

if __name__ == "__main__":

    workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

    print("\n=== WORKFLOW STEPS ===")
    for step in workflow_steps:
        print("-", step)

    completed_steps = []

    for step in workflow_steps:
        print("\n-----------------------------")
        print("Processing step:")
        print(step)

        result = routing_agent.route(step)
        completed_steps.append(result)

        print("\nResult:")
        print(result)

    print("\n=== FINAL OUTPUT ===")
    print(completed_steps[-1])


with open("phase_2/final_output.txt", "w", encoding="utf-8") as f:
    f.write(completed_steps[-1])