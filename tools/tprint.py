#!/usr/bin/env python -u 
'''
pDMET: Density Matrix Embedding theory for Periodic Systems
Copyright (C) 2018 Hung Q. Pham. All Rights Reserved.
A few functions in pDMET are modifed from QC-DMET Copyright (C) 2015 Sebastian Wouters

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Email: Hung Q. Pham <pqh3.14@gmail.com>
'''

import os, datetime
from pDMET import pdmet
import sys
            
def print_header():  
    print("-----------------------------------------------------------------") 
    print("   pDMET: Density Matrix Embedding Theory for Periodic Systems")
    print("                            Version: %s" % (pdmet.__version__))        
    print("                         Author: Hung Q. Pham")
    print("                       Email: pqh3.14@gmail.com")    
    print("                 Current time: %s" % (datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S")))    
    print("-----------------------------------------------------------------")
    sys.stdout.flush()
    
def print_msg(msg=None, *kargs):
    if msg is None:
        print() 
    else:
        print(msg, *kargs)  
    sys.stdout.flush()