import grequests
import requests
from requests.adapters import HTTPAdapter
from threading import Thread
import sys
from Queue import Queue
import numpy as np
import time


def sync_db():
    header = {
        "Authorization": "Basic: a2RsYW5ub3lAZ21haWwuY29tOmZlM2Y2ZDI5OGJlMWI2ODljNmUwZjlkNjFiYjNjY2YzYTNkYWIwMDdmYjYzZWU0MDcxMTFhMTgzMjNjYWQwNzAyNjM5OTY1OTZhOTAwZTM4MzgwNDhhMThjODdkZDUyOWZiZWM3YTA2YTEwZjA0ZDM0NjJjYmRmNjkwNGJlMjEz"}

    urls = [
        'http://tw06v033.ugent.be/Chronic/rest/DrugService/drugs',
        'http://tw06v033.ugent.be/Chronic/rest/SymptomService/symptoms',
        'http://tw06v033.ugent.be/Chronic/rest/TriggerService/triggers',
        'http://tw06v033.ugent.be/Chronic/rest/HeadacheService/headaches?patientID=6',
        'http://tw06v033.ugent.be/Chronic/rest/MedicineService/medicines?patientID=6',
    ]

    session = requests.Session()
    # session.mount('http://', HTTPAdapter(pool_connections=250, pool_maxsize=50))
    rs = (grequests.get(u, headers=header, session=session) for u in urls)

    # responses = requests.async.imap(rs, size=250)

    times = []
    for response in grequests.imap(rs, size=1):
        if response.status_code == 200:
            times.append(response.elapsed.total_seconds())
        else:
            times.append(1000)

        response.close()

    q.put(sum(times))


def send_headache_to_db():
    pass


thread_measures = {1: [], 5: [], 10: [], 25: [], 50: [], 100: [], 250: [], 500: [], 1000: [], 2500: [], 5000: [],
                   10000: []}
n_measures = 1

for i in range(n_measures):
    for key in sorted(thread_measures.keys()):
        # Do the request and add their response times to a queue
        q = Queue(key)
        for i in range(key):
            t = Thread(target=sync_db)
            t.daemon = True
            t.start()

        # Empty the queue and calculate mean of response times, add it to right thread number
        response_times = []
        for items in range(key):
            _time = q.get()
            response_times.append(_time)

        thread_measures[key].append(np.mean(response_times))
        time.sleep(0.5)

print '-------------------------- STRESS TEST: SYNCDB() -----------------------------------'
for key in sorted(thread_measures.keys()):
    print '\t NUMBER OF THREADS = ' + str(key) + '\t AVG TIME = ' + str(np.mean(thread_measures[key]))
