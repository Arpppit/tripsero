from datetime import date

from pandas.io import json
from attractions_recc import *
import pandas as pd
import re
import datetime as dt
import json

low = 50
high = 899
province = 'british_columbia'
start = dt.date.today()
end = start + dt.timedelta(6)
att_df = pd.read_json('etl/attractions.json',orient='records')


category_df = att_df.groupby('category').size().reset_index().sort_values([0],ascending=False)[:20]
categories = list(category_df.category.values)
cat_rating = {'luxury_&_special_occasions': 5.0, 'outdoor_activities': 4.0, 'recommended_experiences': 5.0, 'food,_wine_&_nightlife': 4.0, 'cruises,_sailing_&_water_tours': 3.0, 'cultural_&_theme_tours': 5.0}


def feed_input(price_low,price_high,province,days,att_df,categories,cat_rating):
    filename, user, rbm_att = get_recc(att_df, cat_rating)
    with_url = filter_df(filename, user, low, high, province, att_df)

    final = dict()
    final['timeofday'] = []
    final['image'] = []
    final['name'] = []
    final['location'] = []
    final['price'] = []
    final['rating'] = []
    final['category'] = []
    total_days = end - start
    for i in range(1,total_days.days+2):
        for j in range(2):
            final['timeofday'].append('Morning')
        for j in range(2):
            final['timeofday'].append('Evening')

    for i in range(len(final['timeofday'])): 
        if i%4 == 0: 
            final = top_recc(with_url, final)
        else:
            final = find_closest(with_url, final['location'][-1],final['timeofday'][i], final)


    #print(final)
    t = final_output(days,final)
    return json.dumps(t)

