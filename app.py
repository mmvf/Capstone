import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import ast
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import pickle
from uuid import uuid4
from CustomTransformer import ConvertDataTypes, MergeCategories, FillNA
from sklearn.pipeline import Pipeline

########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    prediction = FloatField()
    predicted_class = IntegerField(null=True)
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

class Error(Model):
    observation_id = TextField(unique=True)
    observation = TextField()

    class Meta:
        database = DB
        
DB.create_tables([Prediction], safe=True)
DB.create_tables([Error], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model

    
with open('columns.json') as fh:
    columns = json.load(fh)


with open('pipeline.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)


with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions


def check_request(request):
    '''
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, Description of the error occurred otherwise
    '''
    
    if 'observation_id' not in request:
        error = 'Field "observation_id" missing from request: {}'.format(request)
        return False, error
    
    return True, ''

def check_valid_column(observation):
    '''
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    '''
    
    valid_columns = {
     'observation_id',
     'Type',
     'Date',
     'Part of a policing operation',
     'Latitude',
     'Longitude',
     'Gender',
     'Age range',
     'Officer-defined ethnicity',
     'Legislation',
     'Object of search',
     'station'
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = 'Missing columns: {}'.format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = 'Unrecognized columns provided: {}'.format(extra)
        return False, error    

    return True, ''

def check_categorical_values(observation):
    '''
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    '''
    
    valid_category_map = {
        'Type': ['Person search', 
                 'Person and Vehicle search', 
                 'Vehicle search'],
        'Gender': ['Male', 
                   'Female', 
                   'Other'],
        'Age range': ['18-24', 
                      'over 34', 
                      '10-17', 
                      '25-34', 
                      'under 10'],
        'Officer-defined ethnicity': ['White', 
                                      'Black', 
                                      'Asian', 
                                      'Other', 
                                      'Mixed']
    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = 'Invalid value provided for {}: {}. Allowed values are: {}'.format(
                    key, value, ','.join(["{}".format(v) for v in valid_categories]))
                return False, error
        else:
            error = 'Categorical field {} missing'
            return False, error

    return True, ''

def check_partofpolice(observation):
    '''
        Validates that observation contains a valid Part of Police Operation value
        
        Returns:
        - assertion value: True if Part of Police Operation  is valid, False otherwise
        - error message: empty if Part of Police Operation  is valid, Error Message otherwise
    '''
      
    part_of_police = observation.get('Part of a policing operation')
        
    if (not pd.isna(part_of_police)) and (type(part_of_police) != bool):
        error = 'Invalid value provided for Part of a policing operation: {}. Allowed values are of type boolean or NaN'.format(part_of_police)
        return False, error
    
    return True, ''

def check_latitude(observation):
    '''
        Validates that observation contains a valid Latitude value
        
        Returns:
        - assertion value: True if Latitude is valid, False otherwise
        - error message: empty if Latitude is valid, Error Message otherwise
    '''
      
    lat = observation.get('Latitude')
    
    if (not pd.isna(lat)) and (not 48 <= lat < 59): 
        error = 'Invalid value provided for Latitude: {}. Allowed values are in the range [48,59[ or NaN.'.format(lat)
        return False, error
    
    return True, ''

def check_longitude(observation):
    '''
        Validates that observation contains a valid Longitude value 
        
        Returns:
        - assertion value: True if Longitude is valid, False otherwise
        - error message: empty if Longitude is valid, Error Message otherwise
    '''
    
    long = observation.get('Longitude')
        
    if (not pd.isna(long)) and not -9 <= long < 3: 
        error = 'Invalid value provided for Longitude: {}. Allowed values are in the range [-9,3[ or NaN.'.format(long)
        return False, error
    
    return True, ''

def check_date(observation):
    '''
        Validates that observation contains a valid Date 
        
        Returns:
        - assertion value: True if Date is valid, False otherwise
        - error message: empty if Date is valid, Error Message otherwise
    '''
    
    date = observation.get('Date')
        
    if type(date) != str: 
        error = 'Invalid value provided for Date: {}. Allowed values are of type string'.format(date)
        return False, error
    
    return True, ''

def check_objectofsearch(observation):
    '''
        Validates that observation contains a valid Object of Search 
        
        Returns:
        - assertion value: True if Object of Search is valid, False otherwise
        - error message: empty if Object of Search is valid, Error Message otherwise
    '''
    
    obj_search = observation.get('Object of search')
        
    if (pd.isna(obj_search)) or (type(obj_search) != str):
        error = 'Invalid value provided for Object of search: {}. Allowed values are of type string'.format(obj_search)
        return False, error
    
    return True, ''

def check_legislation(observation):
    '''
        Validates that observation contains a valid Legislation 
        
        Returns:
        - assertion value: True if Legislation is valid, False otherwise
        - error message: empty if Legislation is valid, Error Message otherwise
    '''
    
    leg = observation.get('Legislation')
        
    if type(leg) != str: 
        error = 'Invalid value provided for Legislation: {}. Allowed values are of type string'.format(leg)
        return False, error
    
    return True, ''

# End input validation functions
########################################

def create_datefeatures(observation):
    
    '''
        Creates month, hour and day_of_week features from Date
        
        Returns:
        - values if feature Date can be read as a date
        - None for all features if Date cannot be read as a date
    '''
    
    date = observation.get('Date')
    
    try:
        date = pd.Timestamp(date)
        hour = date.hour
        month = date.month
        day_of_week = date.day_name()
    except:
        hour = 0
        month = 0
        day_of_week = ''    

    return hour, month, day_of_week


########################################
# Begin webserver stuff

app = Flask(__name__)

@app.route('/should_search/', methods=['POST'])
def predict():
    obs_json = request.get_json()

    #Validate Request
    request_ok, error = check_request(obs_json)
    if not request_ok:
        response = {'error': error}
        return jsonify(response), 405
    
    #Set ID
    _id = obs_json['observation_id']
    observation = obs_json
    
    #Validate Columns present in Request
    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
    
        try:    
            e = Error(observation_id = _id,
                      observation = request.data)
            e.save()
        except IntegrityError:
            error_msg = 'ERROR: Observation ID: "{}" already exists'.format(_id)
            response['error'] = error_msg
            DB.rollback()
            return jsonify(response), 405
        
        return jsonify(response), 405

    #Validate Valid Categories
    categories_ok, error = check_categorical_values(observation)
    if not categories_ok:
        response = {'error': error}
        return jsonify(response), 405

    #Validate Part of a policing operation, if provided
    pop_ok, error = check_partofpolice(observation)
    if not pop_ok:
        response = {'error': error}
        return jsonify(response), 405
    
    #Validate Latitude and Longitude, if provided
    lat_ok, error = check_latitude(observation)
    if not lat_ok:
        response = {'error': error}
        return jsonify(response), 405
    
    long_ok, error = check_longitude(observation)
    if not long_ok:
        response = {'error': error}
        return jsonify(response), 405
    
    #Validate Date
    date_ok, error = check_date(observation)
    if not date_ok:
        response = {'error': error}
        return jsonify(response), 405
    
    #Validate Object of Search
    obj_search_ok, error = check_objectofsearch(observation)
    if not obj_search_ok:
        response = {'error': error}
        return jsonify(response), 405
    
    #Validate Legislation
    leg_ok, error = check_legislation(observation)
    if not leg_ok:
        response = {'error': error}
        return jsonify(response), 405
    
    #Create Date features
    hour, month, day_of_week  = create_datefeatures(observation)  

    #dict in the format need to predict
    obs_df = {'Type': observation.get('Type'),
              'Part of a policing operation': observation.get('Part of a policing operation'),
              'Latitude': observation.get('Latitude'),        
              'Longitude': observation.get('Longitude'),  
              'Gender': observation.get('Gender'),  
              'Age range': observation.get('Age range'),         
              'Officer-defined ethnicity': observation.get('Officer-defined ethnicity'),  
              'Legislation': observation.get('Legislation'),       
              'Object of search': observation.get('Object of search'),      
              'station': observation.get('station'),
              'hour': hour,        
              'month': month,
              'day_of_week': day_of_week}

    #Data Transform
    df_obs = pd.DataFrame([obs_df], columns = columns)
    pipeline_clean = Pipeline([('FillNA', FillNA()),                     
                               ('MergeCategories', MergeCategories()),
                               ('ConvertDataTypes', ConvertDataTypes())])
    
    df_obs = pipeline_clean.fit_transform(df_obs)     
    obs = df_obs.astype(dtypes)

    #Prediction
    proba = pipeline.predict_proba(obs)[0, 1]
    threshold = 0.052
    if proba >= threshold:
        prediction = 1
    else:
        prediction = 0
    
    #prediction = pipeline.predict(obs)[0]
    
    response = {'outcome': bool(prediction)}
    
    p = Prediction(
        observation_id =_id,
        observation = request.data,
        prediction = proba,
        predicted_class = bool(prediction)
    )
    
    try:
        p.save()
        
    except IntegrityError:
        error_msg = 'ERROR: Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        DB.rollback()
        return jsonify(response), 405
    
    return jsonify(response), 200

@app.route('/search_result/', methods=['POST'])
def update():
    obs = request.get_json()
    
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['outcome']
        p.save()
        
        response = {'observation_id': p.observation_id,
                    'outcome': bool(obs['outcome']),
                    'predicted_outcome': bool(p.predicted_class)
                    }
        
        return jsonify(response), 200
    
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg}), 405

@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

# End webserver stuff
########################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
