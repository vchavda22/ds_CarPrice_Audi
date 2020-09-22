#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:14:38 2020
taken the audi data from :
url:  https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes
@author: vikashchavda
"""

import pandas as pd
import numpy as np
df = pd.read_csv('datasets_audi.csv')

# df.head()

df['age'] = df.year.apply(lambda x: x if x < 1 else 2020 - x)

# convert miles to kilometres
df['mileage (km)'] = df.mileage.apply(lambda x: x*1.60934)

# round up the column to integers
df['mileage (km)'] = df['mileage (km)'].round()
# remove the decimal point by converting to integer
df['mileage (km)'] = df['mileage (km)'].astype(int)
# splitting data based on fuel type
df['diesel_yn'] = df['fuelType'].apply(lambda x: 1 if 'diesel' in x.lower() else 0)
df['petrol_yn'] = df['fuelType'].apply(lambda x: 1 if 'petrol' in x.lower() else 0)
df['hybrid_yn'] = df['fuelType'].apply(lambda x: 1 if 'hybrid' in x.lower() else 0)
# splitting data based on transmission
df['manual_yn'] = df['transmission'].apply(lambda x: 1 if 'manual' in x.lower() else 0)
df['semi_automatic_yn'] = df['transmission'].apply(lambda x: 1 if 'semi-auto' in x.lower() else 0)
df['automatic_yn'] = df['transmission'].apply(lambda x: 1 if 'automatic' in x.lower() else 0)

# splitting data based on Car model
df['A_Series_yn'] = df['model'].apply(lambda x: 1 if 'a' in x.lower() else 0)
df['T_Series_yn'] = df['model'].apply(lambda x: 1 if 't' in x.lower() else 0)
df['S_Series_yn'] = df['model'].apply(lambda x: 1 if 's' in x.lower() else 0)
df['R_Series_yn'] = df['model'].apply(lambda x: 1 if 'r' in x.lower() else 0)
df['Q_Series_yn'] = df['model'].apply(lambda x: 1 if 'q' in x.lower() else 0)

# model Type
#df['model_type'] = df['model'].apply(lambda x: 'A' if 'a' in x.lower() or 'Q' if 'q' in x.lower() or 'S' if 's' in x.lower() or 'T' if 't' in x.lower() else 'NA')

df['model_type'] = df['model'].astype(str)
df['model_type'] = df['model'].apply(lambda x: 'A' if 'a' in x.lower() else ('Q' if 'q' in x.lower() else ('R' if 'r' in x.lower() else ('S' if 's' in x.lower() else ('T' if 't' in x.lower() else 'NA'))))) 

df.to_csv('audi_data_cleaned_vc.csv', index=False)