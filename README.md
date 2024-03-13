# MLFinalProject
## Our Goal
We wanted to predict the when the next UFO sighting will be in California and where in California it will take place from latitude and longitude values.

## Our Plan
https://docs.google.com/document/d/1k4_IpIC8IPfEdhbPy6gOxikQjhcO_cDI7SLa1fZwNWQ/edit?usp=sharing

## Running our UI
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


## Git Command Basics

### Make a new branch: 

$ git checkout -b (branchname)

### Clone (duplicate) a branch:

$ git clone -b (branchname) (remote-repo-url)

### Get new changes from github:

Option 1:

$ git fetch #gets new changes but doesn't affect your current changes

Option 2: 

$ git pull #gets new changes and could affect your current changes if conflicts occur

### Upload your changes to github:

note: make sure to first get any new changes before uploading your changes

$ git add (filenameORpathtofile)

$ git status #check if green

$ git commit -m "" #inside quotes can say comment about changes/file contents

$ git push

## Checking Work

Make a pull request  <img width="145" alt="Screenshot 2024-02-10 at 5 07 29 PM" src="https://github.com/priyalpatell/MLFinalProject/assets/93696664/22121220-5ea8-4e79-bdeb-b5c979323274">

From your branch to main branch

Include in description: what your task was and files you changed/added
