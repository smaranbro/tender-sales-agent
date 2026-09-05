import os
from crewai import Agent, LLM

# Define the Groq model
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Pass it to your agent
analyst = Agent(
    role="Tender & RFP Analyst",
    goal="Search and analyze procurement notices",
    backstory="Expert RFP analyst",
    llm=groq_llm,
)