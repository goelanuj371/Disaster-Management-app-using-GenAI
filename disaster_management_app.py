import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate, LLMChain

api_key = 'API Key'  # Replace with your actual API key
llm = GoogleGenerativeAI(model='gemini-pro', google_api_key=api_key)

# Define Prompt Templates
disaster_input_template = """
# Question: What type of disaster would you like to simulate?
# Answer: Please enter the type of disaster you want to simulate (e.g., flood, earthquake, hurricane, wildfire, tornado, blizzard).
# """

scenario_generation_template = """
Question: Generate a detailed {disaster_type} disaster scenario based on historical and predictive data.
Answer: Here is a detailed {disaster_type} disaster scenario:
"""

scenario_summary_template = """
Given the following disaster scenario:
{scenario}
Question: Summarize the key challenges, affected areas, and immediate consequences of this disaster.
Answer: The key challenges, affected areas, and immediate consequences of this disaster are:
"""

prevention_measures_template = """
Given the following summary of the disaster scenario:
{scenario_summary}
Question: Generate tailored prevention and response measures aimed at mitigating the disaster's impact.
Answer: Here are some tailored prevention and response measures to mitigate the disaster's impact:
"""

evacuation_planning_template = """
Given the following disaster scenario and prevention measures:
Disaster Scenario: {scenario}
Prevention Measures: {prevention_measures}
Question: Provide a detailed evacuation plan and identify safe zones for the affected areas.
Answer: Here is a detailed evacuation plan and safe zones for the affected areas:
"""

risk_assessment_template = """
Given the following disaster scenario and prevention measures:
Disaster Scenario: {scenario}
Prevention Measures: {prevention_measures}
Question: Conduct a risk assessment and vulnerability analysis for the disaster.
Answer: Here is a risk assessment and vulnerability analysis for the disaster:
"""

# Define Prompts
disaster_input_prompt = PromptTemplate(template=disaster_input_template, input_variables=[])
scenario_generation_prompt = PromptTemplate(template=scenario_generation_template, input_variables=['disaster_type'])
scenario_summary_prompt = PromptTemplate(template=scenario_summary_template, input_variables=['scenario'])
prevention_measures_prompt = PromptTemplate(template=prevention_measures_template, input_variables=['scenario_summary'])
evacuation_planning_prompt = PromptTemplate(template=evacuation_planning_template, input_variables=['scenario', 'prevention_measures'])
risk_assessment_prompt = PromptTemplate(template=risk_assessment_template, input_variables=['scenario', 'prevention_measures'])

# Define LLM Chains
disaster_input_chain = LLMChain(prompt=disaster_input_prompt, llm=llm)
scenario_generation_chain = LLMChain(prompt=scenario_generation_prompt, llm=llm)
scenario_summary_chain = LLMChain(prompt=scenario_summary_prompt, llm=llm)
prevention_measures_chain = LLMChain(prompt=prevention_measures_prompt, llm=llm)
evacuation_planning_chain = LLMChain(prompt=evacuation_planning_prompt, llm=llm)
risk_assessment_chain = LLMChain(prompt=risk_assessment_prompt, llm=llm)

# Set page title
st.set_page_config(page_title="Disaster Management System", page_icon=":tornado:")

# Set Streamlit app
st.title("Disaster Management System 🌪️ ")
st.markdown("---")

# Disaster Input
st.subheader("Enter Disaster Type")
disaster_type = st.text_input("Enter the type of disaster you want to simulate:")

# Generate Disaster Scenario
if st.button("Generate Scenario"):
    with st.spinner("Generating disaster scenario..."):
        scenario = scenario_generation_chain.run(disaster_type=disaster_type)
        st.session_state.scenario = scenario
    st.success("Disaster scenario generated!")
    st.write(scenario)

# Summarize Disaster Scenario
if 'scenario' in st.session_state:
    if st.button("Summarize Scenario"):
        with st.spinner("Summarizing disaster scenario..."):
            scenario_summary = scenario_summary_chain.run(scenario=st.session_state.scenario)
            st.session_state.scenario_summary = scenario_summary
        st.success("Disaster scenario summary:")
        st.write(scenario_summary)

# Generate Prevention Measures
if 'scenario_summary' in st.session_state:
    if st.button("Generate Prevention Measures"):
        with st.spinner("Generating prevention measures..."):
            prevention_measures = prevention_measures_chain.run(scenario_summary=st.session_state.scenario_summary)
            st.session_state.prevention_measures = prevention_measures
        st.success("Prevention and response measures:")
        st.write(prevention_measures)

# Evacuation Planning
if 'scenario' in st.session_state and 'prevention_measures' in st.session_state:
    if st.button("Evacuation Planning and Safe Zones"):
        with st.spinner("Generating evacuation plan and safe zones..."):
            evacuation_plan = evacuation_planning_chain.run(scenario=st.session_state.scenario, prevention_measures=st.session_state.prevention_measures)
        st.success("Evacuation Plan and Safe Zones:")
        st.write(evacuation_plan)

# Risk Assessment
if 'scenario' in st.session_state and 'prevention_measures' in st.session_state:
    if st.button("Risk Assessment and Vulnerability Analysis"):
        with st.spinner("Conducting risk assessment and vulnerability analysis..."):
            risk_assessment = risk_assessment_chain.run(scenario=st.session_state.scenario, prevention_measures=st.session_state.prevention_measures)
        st.success("Risk Assessment and Vulnerability Analysis:")
        st.write(risk_assessment)

# Restart Button
if st.button("Restart"):
    st.success("Thank you for using the Disaster Management System!")
    # Clear session state to reset the app
    for key in list(st.session_state.keys()):
        del st.session_state[key]
