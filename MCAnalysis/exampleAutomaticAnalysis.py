# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:06:07 2023

@author: massey_j
"""

import pyJM.MCAnalysis as mc
import sys

"""Defined within the script to run from a iPython notebook"""
master = r'/path/to/filesystem'

"Path provided as a variable to run from terminal"
master = sys.argv[1]

mc.automation.analyseAndDistribute(master)