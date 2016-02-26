from headacheclassifier import HeadacheClassifier

__author__ = 'kiani &gilles'

import numpy as np
from numpy.core.records import ndarray
from pymongo import MongoClient
from dateutil.parser import parse
from sklearn.preprocessing import normalize


def loadDataFromDB(host, port, dbname):
    client = MongoClient(host, port)
    db = client[dbname]
    return db


database = loadDataFromDB('localhost', 9000, 'CHRONIC')

# Get all collections from the database
patients = database['patient']
drugs = database['drug']
headaches = database['headache']
medicines = database['medicine']
symptoms = database['symptom']
triggers = database['trigger']


def getNumberOfHeadachesForPatient(patientID):
    number = headaches.find({"patientID": patientID}).count()
    # print "Number of headaches for patient with patientID: %d is %d" % (patientID,number)
    return number, headaches.find({"patientID": patientID})[0:]

def normalized(a, axis=-1, order=2):
    min = np.min(a)
    max = np.max(a)
    newlist = []
    for number in a:
        newlist.append((number - min) / (max - min))
    return newlist


def getLocationMeansForPatient(patientID, normalize=True):
    number, headaches = getNumberOfHeadachesForPatient(patientID)
    mandibular_right = 0.0
    mandibular_left = 0.0
    maxillar_right = 0.0
    maxillar_left = 0.0
    orbital_right = 0.0
    orbital_left = 0.0
    frontal_right = 0.0
    frontal_mid = 0.0
    frontal_left = 0.0
    parietal_right = 0.0
    parietal_mid = 0.0
    parietal_left = 0.0
    temporal_right = 0.0
    temporal_left = 0.0
    occipital_right = 0.0
    occipital_mid = 0.0
    occipital_left = 0.0
    cervical_right = 0.0
    cervical_mid = 0.0
    cervical_left = 0.0

    records = dict((record['_id'], record) for record in headaches)
    for id in records:
        locations = dict((location['key'], location['value']) for location in records[id]['locations'])
        if (locations['mandibular_right']):
            mandibular_right += 1
        if (locations['mandibular_left']):
            mandibular_left += 1
        if (locations['maxillar_right']):
            maxillar_right += 1
        if (locations['maxillar_left']):
            maxillar_left += 1
        if (locations['orbital_right']):
            orbital_right += 1
        if (locations['orbital_left']):
            orbital_left += 1
        if (locations['frontal_right']):
            frontal_right += 1
        if (locations['frontal_mid']):
            frontal_mid += 1
        if (locations['frontal_left']):
            frontal_left += 1
        if (locations['parietal_right']):
            parietal_right += 1
        if (locations['parietal_mid']):
            parietal_mid += 1
        if (locations['parietal_left']):
            parietal_left += 1
        if (locations['temporal_right']):
            temporal_right += 1
        if (locations['temporal_left']):
            temporal_left += 1
        if (locations['occipital_right']):
            occipital_right += 1
        if (locations['occipital_mid']):
            occipital_mid += 1
        if (locations['occipital_left']):
            occipital_left += 1
        if (locations['cervical_right']):
            cervical_right += 1
        if (locations['cervical_mid']):
            cervical_mid += 1
        if (locations['cervical_left']):
            cervical_left += 1
        lijst = np.array(
                [mandibular_right / number, mandibular_left / number, maxillar_right / number, maxillar_left / number,
                 orbital_right / number, orbital_left / number, frontal_right / number, frontal_mid / number,
                 frontal_left / number, parietal_right, parietal_mid / number, parietal_left / number,
                 temporal_right / number, temporal_left / number, occipital_right / number, occipital_mid / number,
                 occipital_left / number, cervical_right / number, cervical_mid,
                 cervical_left / number])
        if (normalize):
            lijst = normalized(lijst)
        return lijst


def getMeanDurationOfHeadachesForPatient(patientID):
    number, headaches = getNumberOfHeadachesForPatient(patientID)
    sum = 0
    records = dict((record['_id'], record) for record in headaches)
    for id in records:
        start = parse(records[id]['intensityValues'][0]['key'])
        end = parse(records[id]['end'])
        if (end < start):
            start = end
        duration = end - start
        sum += duration.seconds / 3600.0
    if number != 0:
        return sum / number
    else:
        return 0


def getMaxMinMeanIntensityValuesForPatient(patientID):
    number, headaches = getNumberOfHeadachesForPatient(patientID)
    max = 0.0
    min = 10.0
    sum = 0.0
    number = 0.0
    records = dict((record['_id'], record) for record in headaches)
    for id in records:
        for value in records[id]['intensityValues']:
            current = value['value']
            if (max < float(current)):
                max = float(current)

            if (min > float(current)):
                min = float(current)

            sum += float(current)
            number += 1.0
    mean = sum / number
    return max, min, mean


def getSymptomIDMinsForPatient(patientID, normalize=True):
    number, headaches = getNumberOfHeadachesForPatient(patientID)
    symptomIDs = symptoms.find()
    symptomIDs = symptomIDs.distinct('symptomID')
    symptomIDs.sort()
    symptom_array = dict((el, 0.0) for el in symptomIDs)
    records = dict((record['_id'], record) for record in headaches)
    for id in records:
        for value in records[id]['symptomIDs']:
            symptom_array[value] += 1
    symptom_array = list(symptom_array.values())
    symptom_array = [x / number for x in symptom_array]
    if normalize:
        symptom_array = normalized(symptom_array)
    return symptom_array


def getTriggerIDMinsForPatient(patientID, normalize=True):
    number, headaches = getNumberOfHeadachesForPatient(patientID)
    triggerIDs = triggers.find()
    triggerIDs = triggerIDs.distinct('triggerID')
    triggerIDs.sort()
    trigger_array = dict((el, 0.0) for el in triggerIDs)
    records = dict((record['_id'], record) for record in headaches)
    for id in records:
        for value in records[id]['triggerIDs']:
            trigger_array[value] += 1
    trigger_array = list(trigger_array.values())
    trigger_array = [x / number for x in trigger_array]
    if normalize:
        trigger_array = normalized(trigger_array)
    return trigger_array


def getFeatureVectorForPatient(patientID):
    # feature vector: [
    #  number of headaches,
    # mean duration of headaches (hours),
    # <array of 1 or 0 for locations of headache (mean per headache)>,
    # max intensityvalue,
    # min intensityvalue,
    # mean intensityvalue,
    # <array of mean symptomids 1 or zero, mean over all headaches for this patient>,
    # <array of mean triggerids 1 or zero, mean over all headaches for this patient>
    # ]

    features = []
    number, niets = getNumberOfHeadachesForPatient(patientID)
    features.append(number)
    features.append(getMeanDurationOfHeadachesForPatient(patientID))
    features.extend(getLocationMeansForPatient(patientID, True))
    features.extend(getMaxMinMeanIntensityValuesForPatient(patientID))
    features.extend(getSymptomIDMinsForPatient(patientID, True))
    features.extend(getTriggerIDMinsForPatient(patientID, True))
    return features

print(getFeatureVectorForPatient(6))
s = HeadacheClassifier()



#
# print "Er zijn %d patienten" % patients.find().count()
# for patient in patients.find():
#     print patient
# test = patients.find_one({"firstName": "Kiani"})
# print test
