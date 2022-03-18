# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 21:43:57 2020

@author: Lilian
"""

from datetime import datetime
import time

time_start = datetime.now()

time.sleep(80);

time_end = datetime.now()

time_interval = time_end - time_start
print(time_interval)