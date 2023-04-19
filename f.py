# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:40:55 2023

@author: ARPITA
"""

from flask import Flask

app=Flask(__name__,template_folder='./templates')

@app.route('/')
def hello():
   print(__name__)
   return 'Arpita'

app.run()
    
