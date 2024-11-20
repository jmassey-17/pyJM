# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:11:03 2024

@author: massey_j-adm
"""

# To do list

import pandas as pd
import datetime
import os
import numpy as np

cols = ['Task', 'Date Added', 'Category', 'Status', "Date Started", 'Date Completed']

wkdir = r'C:\Data\ToDoProject'

file = os.path.join(wkdir, 'ToDoList.csv')

def loadTaskList(file, cols): 
    test = pd.read_csv(file, usecols=(cols))
    test = test.reset_index(drop = True)
    print(f'{wkdir} loaded successfully')
    return test

def newTaskList(cols):
    return pd.DataFrame(columns = cols)


if os.path.exists(file): 
    tasks = loadTaskList(file, cols)
else: 
    tasks = newTaskList(cols)

def saveList(file, tasks=tasks): 
    tasks.to_csv(file)
    return f'{file} saved successfully'

def addTask(task, category, tasks=tasks, file=file): 
    tasks.loc[-1] = [task, datetime.date.today(), category, "Registered", "", ""]  
    tasks.index = tasks.index + 1  
    tasks = tasks.sort_index()  
    saveList(file)
    
def startTask(task, date = datetime.date.today(), tasks=tasks, file=file):
    tasks.loc[tasks['Task'] == task, ['Status', 'Date Started']] = ['Started', date]
    saveList(file)
    
def completeTask(task, date = datetime.date.today(), tasks=tasks, file=file): 
    tasks.loc[tasks['Task'] == task, ['Status', 'Date Completed']] = ['Complete', date]
    saveList(file)
    
def showCategory(category): 
    print(tasks.loc[tasks['Category'] == category, ['Task']])
    
def showAllCategories(): 
    for category in np.unique(tasks['Category']): 
        print(f'tasks for category {category} are:')
        showCategory(category)
    
(tasks['Category'] == category) & ((tasks['Date Completed'] == '') or (tasks['Date Completed'].isna()))

    

