from LLMServer.gcp.claude_instant import ClaudeGcp

model = ClaudeGcp()

prompt = """I am developing python scripts to solve multiple choice exam by RAG system.

This is my current code and I would like to save the question, generated answer and the correct answer in a json file like the format below.

[
  {
    "question": "John signs a contract that has only his wife Mary's name printed on it. Mary was not present when John signed. The contract is for home renovation services. Which of the following statements is most likely to be true regarding the enforceability of this contract?",
    "model_answer": "D",
    "correct_answer": "A",
    "is_correct": false
  },
  {
    "question": "According to the Daubert Standard in the US legal system, which of the following is NOT a criterion for determining the admissibility of expert testimony?",
    "model_answer": "C",
    "correct_answer": "C",
    "is_correct": true
  },
  ]

Your task is:
1. Analyse the current code base
2. Think how we can implement the saving results feature
3. Implement it in the existing codebase
"""
# "question": "An energy company is planning to expand its natural gas processing capabilities in the Panhandle region. Given the information about the existing infrastructure, which strategy would most effectively increase the company's ability to handle varying ethane market conditions while maximizing operational flexibility?\n",

# "choices": [
#       "A) Construct a new processing plant with 200 MMcfd inlet capacity, connected to multiple pipeline systems",
#       "B) Upgrade all existing plants to allow for 24-hour unattended operation",
#       "C) Build additional field compressor stations to increase gathering capacity from producers",
#       "D) Implement a system to dynamically switch between ethane recovery and rejection modes across all plants"
#     ],

# Your task is answer the following:
# 1. What's the correct answer?
# 2. Reasoning
# 3. Based on the Bloom’s taxonomy, which level of question is this?
# 4. Do you need some supporting documents to solve this question?
# 4. How easy is to solve this question with documentation?

# Could you make a multiple choice question and answer with 1 correct and 3 distractors.
# The question must be very difficult and aim for L3 or 4 of Bloom's taxonomy.
# Also the answer is should very difficult to derive without those documentations.

# "documentation": [
#       "During 2012, we completed construction of and placed into service one of the four processing facilities. Phase I expansion of the facility was completed in March bringing inlet capacity to 80 MMcfd. Phase II expansion of the same facility was completed in June bringing the total inlet capacity of the plant to 140 MMcfd. This addition to the Panhandle System enables us to meet our current and expected future processing requirements in this area. We are also improving the connectivity between plants to enable us to better utilize our Panhandle processing capabilities and better serve the growing needs of the area producers, including those in the Granite Wash.\nAll four plants are capable of operating in high ethane recovery mode or in ethane rejection mode and have instrumentation allowing for unattended operation of up to 16 hours per day.\nThe Panhandle System is comprised of a number of pipeline gathering systems and 43 field compressor stations that gather natural gas, directly or indirectly, to the plants. These gathering systems are located in Beaver, Ellis, Harper, and Roger Mills Counties in Oklahoma and Hansford, Hemphill, Hutchinson, Lipscomb, Ochiltree, Roberts and Wheeler Counties in Texas.\nNatural Gas Supply and Markets for Sale of Natural Gas and NGLs. The residue gas from the Antelope Hills plant is delivered into Southern Star Central Gas or Northern Natural Gas pipelines for sale or transportation to market. The NGLs produced at the Antelope Hills plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation to and fractionation at ONEOK’s Conway fractionator.\nThe residue gas from the Beaver plant is delivered into Northern Natural Gas, Southern Star Central Gas or ANR Pipeline Company pipelines for sale or transportation to market. The NGLs produced at the Beaver plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation to and fractionation at ONEOK’s Conway fractionator.\nThe residue gas from the Spearman plant is delivered into Northern Natural Gas or ANR pipelines for sale or transportation to market. The NGLs produced at the Spearman plant are delivered into MAPCO’s (Mid-America Pipeline Company) pipeline system. MAPCO’s pipeline system has the flexibility of delivering the NGLs to either Mont Belvieu or Conway for fractionation.\nThe residue gas from the Sweetwater plant is delivered into Oklahoma Gas Transmission or ANR pipelines for sale or transportation to market. The NGLs produced at the Sweetwater plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation and fractionation, with the majority being handled at ONEOK’s Conway fractionator and a portion being delivered to the Mont Belvieu markets.\nCrescent System\nGeneral. The Crescent System is a natural gas gathering system stretching over seven counties within central Oklahoma’s Sooner Trend. The system consists of approximately 1,724 miles of natural gas gathering pipelines, ranging in size from two to 10 inches in diameter, and the Crescent natural gas processing plant located in Logan County, Oklahoma. Fourteen compressor stations are operating across the Crescent System. We continue to look at potential growth opportunities to service the Mississippian Lime formation.\nThe Crescent plant is a NGL recovery plant with current capacity of approximately 40 MMcfd. The Crescent facility also includes a gas engine-driven generator which is routinely operated, making the plant self-sufficient with respect to electric power. The cost of fuel (residue gas) for the generator is borne by the producers under the terms of their respective gas contracts.",
#       "The Spearman plant has 100 MMcfd of inlet capacity. The plant is capable of operating in high ethane recovery mode or in ethane rejection mode and has instrumentation allowing for unattended operation of up to 16 hours per day.\nThe Sweetwater plant is capable of operating in high ethane recovery mode or in ethane rejection mode and has instrumentation allowing for unattended operation of up to 16 hours per day.\nIn conjunction with the acquisition of the Sweetwater plant, two new gas compressor stations were installed; one is located on the east end of the North Canadian pipeline and the other on the east end of the Hemphill pipeline.\nNatural Gas Supply and Markets for Sale of Natural Gas and NGLs. The supply in the Panhandle System comes from approximately 203 producers pursuant to 332 contracts. The residue gas from the Beaver plant can be delivered into the Northern Natural Gas, Southern Star Central Gas or ANR Pipeline Company pipelines for sale or transportation to market. The NGLs produced at the Beaver plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation to and fractionation at ONEOK’s Conway fractionator.\nThe residue gas from the Spearman plant is delivered into Northern Natural Gas pipelines for sale or transportation to market. The NGLs produced at the Spearman plant are delivered into MAPCO’s (Mid-America Pipeline Company) pipeline system. MAPCO’s pipeline system has the flexibility of delivering the NGLs to either Mont Belvieu or Conway for fractionation.\nThe residue gas from the Sweetwater plant is delivered into Northern Natural Gas pipelines for sale or transportation to market. The NGLs produced at the Sweetwater plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation to and fractionation at ONEOK’s Conway fractionator.\nCrossroads System General. The Crossroads System is a natural gas gathering system located in the southeast portion of Harrison County, Texas. The Crossroads System consists of approximately eight miles of natural gas gathering pipelines, ranging in size from eight to twelve inches in diameter, and the Crossroads plant. The Crossroads System also includes approximately 20 miles of six-inch NGL pipeline that transport the NGLs produced at the Crossroads plant to the Panola Pipeline.\nThe Crossroads plant has 80 MMcfd of inlet capacity. The plant is capable of operating in high ethane recovery mode or in ethane rejection mode and has instrumentation allowing for unattended operation of up to 16 hours per day.\nNatural Gas Supply and Markets for Sale of Natural Gas and NGLs. The natural gas on the Crossroads System originates from the Bethany Field from where we have contracted with five producers. The Crossroads System delivers the residue gas from the Crossroads plant into the CenterPoint Energy pipeline for sale or transportation to market. The NGLs produced at the Crossroads plant are delivered into the Panola Pipeline for transportation to Mont Belvieu, Texas for fractionation.\nCrescent System General. The Crescent System is a natural gas gathering system stretching over seven counties within central Oklahoma’s Sooner Trend. The system consists of approximately 1,701 miles of natural gas gathering pipelines, ranging in size from two to 10 inches in diameter, and the Crescent natural gas processing plant located in Logan County, Oklahoma. Fifteen compressor stations are operating across the Crescent System.\nThe Crescent plant is a NGL recovery plant with current capacity of approximately 40 MMcfd. The Crescent facility also includes a gas engine-driven generator which is routinely operated, making the plant self-sufficient with respect to electric power. The cost of fuel (residue gas) for the generator is borne by the producers under the terms of their respective gas contracts."
#     ]

print(model.invoke(prompt))
