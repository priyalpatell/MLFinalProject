# MLFinalProject
## Our Goal
We wanted to predict the when the next UFO sighting will be in California and where in California it will take place using latitude and longitude values.

## View Our Code
Importing Keras tends to crash Jupyter Notebook, so please only view our code from this google colab link which seemed to fix this problem
https://colab.research.google.com/drive/1LkhfuaI1NOLcgpFoup_GG5OF2Zsy6dzQ?usp=sharing

Note: you will be instructed to log into you google drive account associated with your ucdavis email & select ALL for access
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

## Our Demo
https://drive.google.com/file/d/1fa86uAFri8edmpGrIdrGIFHJqRP_1g2J/view?usp=sharing

## Todo
- update roadmap
- update research section
- add access to google colab
- upload new code to github
- double check everything

### Git Clone Remote Respository Error
Error: If Git - remote: Repository not found

Solution: Generate a personal access token in github settings > developer settings > personal access tokens

Add personal access token to your clone https url link

$ git clone https://(personalaccesstoken)github.com/(your username)/MLFinalProject.git
