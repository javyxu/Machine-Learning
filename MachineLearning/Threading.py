# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170504
# Email: xujavy@gmail.com
# Description: Threading
##########################

import threading
import time

def worker(num):
    ''' Thread worker function
    '''

    time.sleep(1)
    print('The number is %d\n' % num)
    return

for i in range(20):
    t = threading.Thread(target=worker, args=(i,),
                        name='t.%d' % i)
    t.start()
