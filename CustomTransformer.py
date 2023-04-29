from sklearn.base import BaseEstimator, TransformerMixin # to create classes
from sklearn.compose import ColumnTransformer

class ConvertDataTypes(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, data):
        X = data.copy()

        # Convert Columns with Category Values
        cat = ['Type',        
               'Part of a policing operation',   
               'Gender',      
               'Age range',   
               'Officer-defined ethnicity',  
               'Legislation',    
               'Object of search',
               'station',      
               'day_of_week']
        
        for c in cat:
            X[c] = X[c].apply(lambda x: str(x).lower())

        X[cat] = X[cat].astype('category')

        return X

class MergeCategories(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X=None, y=None, **fit_params):
        return self 

    def transform(self, data):
        X = data.copy()

        # Merge Age Range
        X['Age range'] = X['Age range'].replace('under 10', 'under 18')
        X['Age range'] = X['Age range'].replace('10-17', 'under 18')

        prevalent_legislation = ['Police and Criminal Evidence Act 1984 (section 1)',
                                 'Misuse of Drugs Act 1971 (section 23)',
                                 'Firearms Act 1968 (section 47)',
                                 'Criminal Justice and Public Order Act 1994 (section 60)', 
                                 'unknown']

        X['Legislation'] = X['Legislation'].apply(lambda x: 'other' if x not in prevalent_legislation else x)

        return X

class FillNA(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, data):
        X = data.copy()

        # Impute Missing Values Lat and Long
        coordinate_dict = {'Latitude': {'avon-and-somerset': 51.33175637267479, 
                           'bedfordshire': 51.98144995771613, 
                           'btp': 52.12218768654392, 
                           'cambridgeshire': 52.406997233214504, 
                           'cheshire': 53.27632442323492, 
                           'city-of-london': 51.515312816717575, 
                           'cleveland': 54.34701910875864, 
                           'cumbria': 54.56496815974666, 
                           'derbyshire': 53.00366546443391, 
                           'devon-and-cornwall': 50.5275507427191, 
                           'dorset': 50.72136650800582, 
                           'durham': 54.68126312370846, 
                           'dyfed-powys': 52.12028764514218,
                           'essex': 51.72260861278373, 
                           'gloucestershire': 51.850543872809496, 
                           'hampshire': 50.93112387363688, 
                           'hertfordshire': 51.768994971696216, 
                           'humberside': 53.701766442985075, 
                           'kent': 51.39447867441328, 
                           'lancashire': 53.17557931559899, 
                           'leicestershire': 52.65100891407568, 
                           'lincolnshire': 53.07508733459697, 
                           'merseyside': 53.42796231526082, 
                           'metropolitan': 51.507594764548436, 
                           'norfolk': 52.64004391303848, 
                           'north-wales': 53.180918865456725, 
                           'north-yorkshire': 54.064504035898906, 
                           'northamptonshire': 52.3072518685, 
                           'northumbria': 55.00710826101568, 
                           'nottinghamshire': 51.92553576029973, 
                           'south-yorkshire': 51.92553576029973, 
                           'staffordshire': 52.90053738474921, 
                           'suffolk': 52.15084482566662, 
                           'surrey': 51.325869422737, 
                           'sussex': 50.902363490320866, 
                           'thames-valley': 51.68988934877214, 
                           'warwickshire': 52.37148753310205,
                           'west-mercia': 52.41016790308912, 
                           'west-midlands': 52.48964841613634, 
                           'west-yorkshire': 53.76984142554578, 
                           'wiltshire': 51.421036072453624}, 
                   'Longitude': {'avon-and-somerset': -2.7071552408137083, 
                            'bedfordshire': -0.426417996229464, 
                            'btp': -0.8921149511598153, 
                            'cambridgeshire': -0.0774420258467023, 
                            'cheshire': -2.612427713992298, 
                            'city-of-london': -0.08623273007497918, 
                            'cleveland': -1.2172083120124806,
                            'cumbria': -3.1320293015482052, 
                            'derbyshire': -1.4851611348547717, 
                            'devon-and-cornwall': -4.026554344383304, 
                            'dorset': -2.043267481441048, 
                            'durham': -1.581104200502653, 
                            'dyfed-powys': -4.007564165580568, 
                            'essex': 0.5533465492591425, 
                            'gloucestershire': -2.1959625771622386, 
                            'hampshire': -1.2145474441259134, 
                            'hertfordshire': -0.2599600221595488, 
                            'humberside': -0.4027265295522388, 
                            'kent': 0.5128513800228963, 
                            'lancashire': -2.6421476185305828, 
                            'leicestershire': -1.1536536537719932, 
                            'lincolnshire': -0.31962162829209895, 
                            'merseyside': -2.942090273742405, 
                            'metropolitan': -0.1078484889388174, 
                            'norfolk': 0.9602396567489948, 
                            'north-wales': -3.5553297378131092, 
                            'north-yorkshire': -1.11896528058587, 
                            'northamptonshire': -0.8103302245, 
                            'northumbria': -1.5863827457057507,
                            'nottinghamshire': -0.6653726503038662, 
                            'south-yorkshire': -0.6653726503038662, 
                            'staffordshire': -2.0699261203247317, 
                            'suffolk': 1.020457451894589, 
                            'surrey': -0.4328004846962508, 
                            'sussex': -0.16636629196499603, 
                            'thames-valley': -0.9966845383898041, 
                            'warwickshire': -1.4922003654235327, 
                            'west-mercia': -2.3419879800437395, 
                            'west-midlands': -1.8710733354546436, 
                            'west-yorkshire': -1.648371743890518, 
                            'wiltshire': -1.9036338988449422}}

        try:
            X['Latitude'] = X['Latitude'].fillna(X['station'].map(coordinate_dict['Latitude'])).values
        except KeyError:
            X['Latitude'] = 52.3
        try:
            X['Longitude'] = X['Longitude'].fillna(X['station'].map(coordinate_dict['Longitude'])).values
        except KeyError:
            X['Longitude'] = -1.3
            
        #Fill NA for Legislation  and Part of a policing operation
        X['Legislation'] = X['Legislation'].fillna('unknown')
        
        X[['Part of a policing operation']] = X[['Part of a policing operation']].fillna(False).astype('boolean')

        return X

