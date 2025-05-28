import warnings
warnings.filterwarnings("ignore")

from crewai import Agent, Task, Crew


import os 

os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

support_agent = Agent(
    role="Senior Support Representative",
    goal="Be the most friendly and helpful "
    "support representatitve in your teams",
    backstory=(
        "You work at crewAI (https://crewai.com) and "
        " are now working on providing "
        "support to {customer}, a super important customer "
        " for your company."
        "You need to make sure that you proide the best support!"
        "Make sure to provide full complete answers, "
        " and make o assumptions."
        ),
        allow_delegation=False,
        verbose=True) 

support_quality_agent = Agent(
    role="Support Quality Assurance Specialist",
    goal="Get recognition for providing the "
        "best support quality assurance in your team",
    backstory=(
        "You work at crewAI (https://crewai.com) and "
         "are now woking in your team "
         "on a request from {customer} ensuring that "
         "the support representative is "
         "providing th best support possible.\n"
         "You need to make sure that the suppot representative "
         "is providing full complete answers, and make no assumptions.  "
         ),
         verbose=True
         )

from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool
                        
# Possible Custom Tools
## Load customer data
## Tap into previous conversations
## Load data from a CRM
## Checking existing bug reports
## Checking existing feature requests
## Checking ongoing tickets
## and more ...

search_tool = SerperDevTool()

scrape_tool = ScrapeWebsiteTool()

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/hot-to/Creating-a-Crew-and-kick-it-off/"
)

# Different Ways to Give Agents Tools 

## Agent Level: The Agent can use Tool(s) on any Task it performs.
## Task LEvel: The Agent will only use the Tool(s) when performing that specific task.

# Note: Task Tools override the Agent Tools.

inquery_resolution= Task(
    description=(
    "{customer} just reached out with a super important ask:\n"
    "{inquery}\n\n"
    "{person} from {customer is the one that reached out. "
    "Make sure to use everything you know "
    "to provide the best support possible."
    "You ust strive to provide a complete "
    "and accurate response to the customer's inquiry."
    ),
    expected_output=(
        "A detailed, infomative response to the "
        "customer's inquiry that addresses "
        "all aspects of ther question.\n"
        "The response should include references "
        "to everything yo used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer in complete, "
        "leaving no questions unanswered, and maintain a helpul and and friendly "
		"tone throughout."
    ),
	tools=[docs_scrape_tool],
    agent=support_agent,
)

## quality_assurance_review is not using any Tool(s)
## Here the QA Agent will only review the work of the Support Agent


quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
		"high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry "
        "have been addressed "
		"thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to "
        " find the information, "
		"ensuring the response is well-supported and "
        "leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response "
        "ready to be sent to the customer.\n"
        "This response should fully address the "
        "customer's inquiry, incorporating all "
		"relevant feedback and improvements.\n"
		"Don't be too formal, we are a chill and cool company "
	    "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent,
)

# Setting memory=True when putting the crew together enables Memory.

crew = Crew(
  agents=[support_agent, support_quality_assurance_agent],
  tasks=[inquiry_resolution, quality_assurance_review],
  verbose=2,
  memory=True
)

#Guardrails
#By running the execution below, you can see that the agents and the 
#responses are within the scope of what we expect from them.

inputs = {
    "customer": "DeepLearningAI",
    "person": "Andrew Ng",
    "inquiry": "I need help with setting up a Crew "
               "and kicking it off, specifically "
               "how can I add memory to my crew? "
               "Can you provide guidance?"
}
result = crew.kickoff(inputs=inputs)

from IPython.display import Markdown
Markdown(result)