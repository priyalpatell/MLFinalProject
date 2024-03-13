# MLFinalProject
## Our Goal
We wanted to predict the when the next UFO sighting will be in California and where in California it will take place using latitude and longitude values.

## Our Plan
https://docs.google.com/document/d/1k4_IpIC8IPfEdhbPy6gOxikQjhcO_cDI7SLa1fZwNWQ/edit?usp=sharing

## View Our Code
Importing Keras Tends to Crash Jupyter Notebook, so please view our code from this google colab link instead
https://colab.research.google.com/drive/1LkhfuaI1NOLcgpFoup_GG5OF2Zsy6dzQ?usp=sharing

## Running Our UI
$ streamlit run project_code_ui.py

Note: takes a bit of time to load results

Example:

  Input of year = 1
  
    question 1 ~8 minutes
    
    question 2 ~9 minutes
    
  Input of year = 2
  
    question 1 ~16 minutes
    
    question 2 ~9 minutes

### Git Clone Remote Respository Error
Error: If Git - remote: Repository not found

Solution: Generate a personal access token in github settings > developer settings > personal access tokens

Add personal access token to your clone https url link

$ git clone https://(personalaccesstoken)github.com/(your username)/MLFinalProject.git
