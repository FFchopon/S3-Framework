from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver
from langchain_quickjs import CodeInterpreterMiddleware

backend = StateBackend()
checkpointer = MemorySaver()

# -------------------------
# Load skill
# -------------------------

skill_content = Path("skills/security-skill/SKILL.md").read_text(encoding="utf-8")
script_content = Path("skills/security-skill/security_check.py").read_text(encoding="utf-8")

skills_files = {
    "/skills/security-skill/SKILL.md": create_file_data(skill_content),
    "/skills/security-skill/security_check.py": create_file_data(script_content),
}

agent = create_deep_agent(
    model="openai:gpt-5-mini",
    backend=backend,
    skills=["/skills/"],
    checkpointer=checkpointer,

    # VERY IMPORTANT
    middleware=[
        CodeInterpreterMiddleware(
            skills_backend=backend
        )
    ],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "An AI agent executed a suspicious script. "
                    "Please help investigate."
                ),
            }
        ],
        "files": skills_files,
    },
    config={"configurable": {"thread_id": "demo"}},
)

print("\n====================\n")
print(result["messages"][-1].content)
print("\n====================\n")