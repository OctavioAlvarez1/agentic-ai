# 🤖 AI-Powered Agentic Workflow for Project Management

> **Enterprise-style Proof of Concept (PoC)** demonstrating how autonomous AI agents can collaborate to dynamically plan, validate, and execute complex workflows using agentic architectures.

---

## 📌 Project Overview

This project implements a **multi-agent AI system** that simulates how a **Technical Program Manager (TPM)** coordinates specialized agents to transform a high-level product idea into:

- ✅ User personas & user stories  
- ✅ Product features  
- ✅ Detailed engineering tasks  

The system showcases **agentic workflows**, where multiple AI agents reason independently, evaluate each other’s outputs, and adapt dynamically — rather than following rigid, pre-defined automation steps.

This project was developed as part of an **advanced Agentic AI engineering program** and is structured in two phases:

- **Phase 1:** Agent library design & validation  
- **Phase 2:** End-to-end agentic workflow orchestration  

---

## 🧠 Key Concepts Demonstrated

- Agent-based system design  
- Prompt engineering with personas and constraints  
- Knowledge-augmented generation  
- Evaluation & self-correction loops  
- Retrieval-Augmented Generation (RAG)  
- Embedding-based routing & semantic similarity  
- Action planning and workflow orchestration  

---

## 🏗️ Architecture Overview

### Agent Roles

| Agent | Responsibility |
|------|----------------|
| **DirectPromptAgent** | Basic LLM interaction (baseline) |
| **AugmentedPromptAgent** | Persona-driven responses |
| **KnowledgeAugmentedPromptAgent** | Strict knowledge-constrained answers |
| **EvaluationAgent** | Validates and iteratively improves responses |
| **RoutingAgent** | Routes tasks using semantic similarity |
| **ActionPlanningAgent** | Extracts ordered workflow steps |
| **RAGKnowledgePromptAgent** | Retrieves answers from embedded knowledge |

Agents interact dynamically to produce structured outputs that resemble real-world product planning artifacts.

---

## 📁 Repository Structure
```
starter/
├── phase_1/
│   ├── workflow_agents/
│   │   ├── base_agents.py
│   │   └── __init__.py
│   ├── direct_prompt_agent.py
│   ├── augmented_prompt_agent.py
│   ├── knowledge_augmented_prompt_agent.py
│   ├── evaluation_agent.py
│   ├── routing_agent.py
│   ├── action_planning_agent.py
│   └── rag_knowledge_prompt_agent.py
│
├── phase_2/
│   ├── workflow_agents/
│   │   └── __init__.py
│   ├── agentic_workflow.py
│   ├── Product-Spec-Email-Router.txt
│   └── output_agentic_workflow.txt
│
├── tests/
│   └── .env
├── requirements.txt
└── README.md
```
## 🧪 Testing & Validation

Each agent includes a standalone executable test script that demonstrates:

-  Prompt used
- Agent response
- Required explanation or confirmation statements

Validated agents:

- DirectPromptAgent
- AugmentedPromptAgent
- KnowledgeAugmentedPromptAgent
- EvaluationAgent
- RoutingAgent
- ActionPlanningAgent
- RAGKnowledgePromptAgent

All test outputs were executed locally and verified.

## 🛠️ Tech Stack

- Python 3.12
- OpenAI API
- GPT-3.5-Turbo
- text-embedding-3-large
- NumPy (cosine similarity)
- Pandas (RAG storage)
- dotenv (secure config)

## 🎯 Key Concepts Demonstrated

- Agentic workflows vs traditional pipelines
- Prompt engineering by agent role
- Semantic routing using embeddings
- Iterative evaluation loops
- Retrieval-augmented generation (RAG)
- Modular, extensible agent libraries

## 📌 Example Use Cases

- Product & project planning automation
- AI-assisted TPM workflows
- Enterprise knowledge orchestration
- LLM-based decision systems
- Multi-agent AI architectures

## 📈 Why This Matters

This project goes beyond chatbots.
It demonstrates how AI agents can collaborate, reason, validate, and improve outputs, mirroring real enterprise workflows.

It’s a foundation for:

- Scalable AI systems
- Autonomous project planning
- Intelligent orchestration layers

## 👤 Author

Octavio Alvarez
Senior Data / AI Engineer
Focused on agentic systems, data platforms, and enterprise AI architectures.
