from pymongo import MongoClient


class DataCollector(object):

    def __init__(self):
        pass

    @staticmethod
    def load_data_from_db(host, port, dbname):
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