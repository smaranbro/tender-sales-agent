import os
import resend
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# 1. Initialize Credentials
GROQ_KEY = os.getenv("GROQ_API_KEY")
SERPER_KEY = os.getenv("SERPER_API_KEY")
RESEND_KEY = os.getenv("RESEND_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "onboarding@resend.dev")
TARGET_RECIPIENT = os.getenv("TARGET_RECIPIENT")

# Validate required keys
if not GROQ_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Add it to GitHub Secrets.")
if not SERPER_KEY:
    raise ValueError("SERPER_API_KEY environment variable is not set.")
if not TARGET_RECIPIENT:
    raise ValueError("TARGET_RECIPIENT environment variable is not set.")

resend.api_key = RESEND_KEY

# 2. Configure Groq LLM & Web Tools
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=GROQ_KEY
)

search_tool = SerperDevTool(api_key=SERPER_KEY)
scrape_tool = ScrapeWebsiteTool()

# 3. Define Specialized Agents
tender_researcher = Agent(
    role="Tender & RFP Analyst",
    goal="Discover active public tenders, RFPs, and contract opportunities in target industries.",
    backstory="You specialize in scanning procurement notices, government portals, and industry boards for open bids.",
    tools=[search_tool, scrape_tool],
    llm=llm,
    verbose=True
)

contact_finder = Agent(
    role="Lead Investigator",
    goal="Locate contact emails and procurement lead details for targeted tender opportunities.",
    backstory="You search public web directories, company press pages, and contact listings to find verified email addresses.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

pitch_copywriter = Agent(
    role="Outbound Sales Strategist",
    goal="Craft short, compelling cold outreach emails tailored to specific tender requirements.",
    backstory="You write high-converting B2B emails that directly align company capabilities with client RFP requirements.",
    llm=llm,
    verbose=True
)

# 4. Define Tasks
task_find_tenders = Task(
    description=(
        "Search for active RFPs, procurement notices, or software/consulting tenders posted in the past 30 days. "
        "Extract project title, issuing organization, scope summary, and web URL."
    ),
    expected_output="A structured summary of the best active tender found.",
    agent=tender_researcher
)

task_find_contacts = Task(
    description=(
        "For the tender found in the previous step, search for the procurement office or contact email address."
    ),
    expected_output="Organization name, contact person/department, and verified email address.",
    agent=contact_finder
)

task_draft_pitch = Task(
    description=(
        "Write a cold sales email (under 150 words) targeting the contact person. "
        "Mention the specific tender title, highlight relevant solutions, and ask for a intro call."
    ),
    expected_output="An email formatted with Subject Line and Body text.",
    agent=pitch_copywriter
)

# 5. Execute Agent Crew
def run_pipeline():
    crew = Crew(
        agents=[tender_researcher, contact_finder, pitch_copywriter],
        tasks=[task_find_tenders, task_find_contacts, task_draft_pitch]
    )

    result = crew.kickoff()
    pitch_text = str(result)
    
    print("\n--- AGENT OUTPUT ---")
    print(pitch_text)

    # 6. Send Email via Resend API
    try:
        email_payload = {
            "from": SENDER_EMAIL,
            "to": [TARGET_RECIPIENT],
            "subject": "New Tender Opportunity Lead & Pitch Draft",
            "html": f"<pre style='font-family: sans-serif;'>{pitch_text}</pre>"
        }
        response = resend.Emails.send(email_payload)
        print(f"\n[Success] Pitch email dispatched! ID: {response['id']}")
    except Exception as e:
        print(f"\n[Error] Email dispatch failed: {e}")

if __name__ == "__main__":
    run_pipeline()
