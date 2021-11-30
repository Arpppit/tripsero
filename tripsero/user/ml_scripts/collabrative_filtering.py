from datetime import date

from pandas.io import json
import sys
#from attractions_recc import *
import pandas as pd
import re
import datetime as dt
import json

low = 50
high = 899
province = 'british_columbia'
start = dt.date.today()
end = start + dt.timedelta(6)










import pandas as pd
import numpy as np
# import ipywidgets as w
# from ipywidgets import HBox, VBox
# from ipywidgets import Layout, widgets
# from IPython.display import display, IFrame, HTML
from utils import Util
from rbm import RBM
import math, re, datetime as dt, glob
from urllib.parse import quote
from urllib.request import Request, urlopen
#from google_images_download import google_images_download
from PIL import Image
import requests
from bs4 import BeautifulSoup
import html5lib
#from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
def f(row):
    avg_cat_rat = dict()
    for i in range(len(row['category'])):
        if row['category'][i] not in avg_cat_rat:
            avg_cat_rat[row['category'][i]] = [row['rating'][i]]
        else:
            avg_cat_rat[row['category'][i]].append(row['rating'][i])
    for key,value in avg_cat_rat.items():
        avg_cat_rat[key] = sum(value)/len(value)
    return avg_cat_rat

def sim_score(row):
    score = 0.0
    match = 0
    col1 = row['cat_rat']
    col2 = row['user_data']
    for key, value in col2.items():
        if key in col1:
            match+=1
            score += (value-col1[key])**2
    if match != 0:
        return ((math.sqrt(score)/match) + (len(col2) - match))
    else:
        return 100

def get_recc(att_df, cat_rating):
    util = Util()
    epochs = 50
    rows = 40000
    alpha = 0.01
    H = 128
    batch_size = 16
    dir= 'etl/'
    ratings, attractions = util.read_data(dir)
    ratings = util.clean_subset(ratings, rows)
    rbm_att, train = util.preprocess(ratings)
    num_vis =  len(ratings)
    rbm = RBM(alpha, H, num_vis)
    
    joined = ratings.set_index('attraction_id').join(attractions[["attraction_id", "category"]].set_index("attraction_id")).reset_index('attraction_id')
    grouped = joined.groupby('user_id')
    category_df = grouped['category'].apply(list).reset_index()
    rating_df = grouped['rating'].apply(list).reset_index()
    cat_rat_df = category_df.set_index('user_id').join(rating_df.set_index('user_id'))
    cat_rat_df['cat_rat'] = cat_rat_df.apply(f,axis=1)
    cat_rat_df = cat_rat_df.reset_index()[['user_id','cat_rat']]
    
    cat_rat_df['user_data'] = [cat_rating for i in range(len(cat_rat_df))]
    cat_rat_df['sim_score'] = cat_rat_df.apply(sim_score, axis=1)
    user = cat_rat_df.sort_values(['sim_score']).values[0][0]
    
    print("Similar User: {u}".format(u=user))
    filename = "e"+str(epochs)+"_r"+str(rows)+"_lr"+str(alpha)+"_hu"+str(H)+"_bs"+str(batch_size)
    reco, weights, vb, hb = rbm.load_predict(filename,train,user)
    unseen, seen = rbm.calculate_scores(ratings, attractions, reco, user)
    rbm.export(unseen, seen, 'recommendations/'+filename, str(user))
    return filename, user, rbm_att

def filter_df(filename, user, low, high, province, att_df):
    recc_df = pd.read_csv('recommendations/'+filename+'/user{u}_unseen.csv'.format(u=user), index_col=0)
    recc_df.columns = ['attraction_id', 'att_name', 'att_cat', 'att_price', 'score']
    recommendation = att_df[['attraction_id','name','category','city','latitude','longitude','price','province', 'rating']].set_index('attraction_id').join(recc_df[['attraction_id','score']].set_index('attraction_id'), how="inner").reset_index().sort_values("score",ascending=False)
    
    filtered = recommendation[(recommendation.province == province) & (recommendation.price >= low) & (recommendation.price >= low)]
    url = pd.read_json('outputs/attractions_cat.json',orient='records')
    url['id'] = url.index
    with_url = filtered.set_index('attraction_id').join(url[['id','attraction']].set_index('id'), how="inner")
    print(with_url.head())
    return with_url

def get_image(name):
  url = url =f'https://www.google.com/search?q={name}&hl=en-GB&source=lnms&tbm=isch&sa=X&ved=2ahUKEwi77e_zg_zzAhU64zgGHWyiCYgQ_AUoA3oECAEQBQ&biw=1920&bih=1007'
  res = requests.get(url)
  bs =BeautifulSoup(res.content, 'html5lib')
  table = bs.find_all('img')
  if len(table) >=6: 
    return table[5].get('src')
  else:
    return table[1].get('src')


# def get_image(name):
#     name = name.split(",")[0]
#     response = google_images_download.googleimagesdownload()
#     args_list = ["keywords", "keywords_from_file", "prefix_keywords", "suffix_keywords",
#              "limit", "format", "color", "color_type", "usage_rights", "size",
#              "exact_size", "aspect_ratio", "type", "time", "time_range", "delay", "url", "single_image",
#              "output_directory", "image_directory", "no_directory", "proxy", "similar_images", "specific_site",
#              "print_urls", "print_size", "print_paths", "metadata", "extract_metadata", "socket_timeout",
#              "thumbnail", "language", "prefix", "chromedriver", "related_images", "safe_search", "no_numbering",
#              "offset", "no_download"]
#     args = {}
#     for i in args_list:
#         args[i]= None
#     args["keywords"] = name
#     args['limit'] = 1
#     params = response.build_url_parameters(args)
#     url = 'https://www.google.com/search?q=' + quote(name) + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' + params + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
#     try:
#         response.download(args)
        
#         for filename in glob.glob("downloads/{name}/*jpg".format(name=name)) + glob.glob("downloads/{name}/*png".format(name=name)):
#             return filename
#     except:
#         for filename in glob.glob("downloads/*jpg"):
#             return filename

def top_recc(with_url, final):
    i=0
    print(with_url)
    print(final)
    try:
        while(1):
            first_recc = with_url.iloc[[i]]
            if(first_recc['name'].values.T[0] not in final['name']):
                final['name'].append(first_recc['name'].values.T[0])
                final['location'].append(first_recc[['latitude','longitude']].values.tolist()[0])
                final['price'].append(first_recc['price'].values.T[0])
                final['rating'].append(first_recc['rating'].values.T[0])
                final['image'].append(get_image(first_recc['name'].values.T[0]))
                #final['image'].append('www.google.com/image')
                final['category'].append(first_recc['category'].values.T[0])
                return final
            else:
                i+=1
    except Exception as e:
        return final

def find_closest(with_url, loc, tod, final):
    syns1 = wordnet.synsets("evening")
    syns2 = wordnet.synsets("night")
    evening = [word.lemmas()[0].name() for word in syns1] + [word.lemmas()[0].name() for word in syns2]
    distance = list()
    for i in with_url[['latitude','longitude']].values.tolist():
        distance.append(math.sqrt((loc[0]-i[0])**2 + (loc[1]-i[1])**2))
    with_dist = with_url
    with_dist["distance"] = distance
    sorted_d = with_dist.sort_values(['distance','price'], ascending=['True','False'])
    if tod == "Evening":
        mask = sorted_d.name.apply(lambda x: any(j in x for j in evening))
    else:
        mask = sorted_d.name.apply(lambda x: all(j not in x for j in evening))
    final = top_recc(sorted_d[mask], final)
    return final

def final_output(days, final):
    time = ['MORNING', 'EVENING']
    fields = ['NAME', 'CATEGORY', 'LOCATION', 'PRICE', 'RATING']
    recommendations = ['Recommendation 1:', 'Recommendation 2:']

    # box_layout = Layout(justify_content='space-between',
    #                     display='flex',
    #                     flex_flow='row', 
    #                     align_items='stretch',
    #                    )
    # column_layout = Layout(justify_content='space-between',
    #                     width='75%',
    #                     display='flex',
    #                     flex_flow='column', 
    #                    )
    tab = {}
    tab['name']=[]
    tab['image']=[]
    tab['price']=[]
    tab['rating']=[]
    tab['category']=[]
    tab['location']=[]
    for i in range(days):
        tab['image'].append(final['image'][i*4:(i+1)*4])
        #images = [open(i, "rb").read() for i in images]
        tab['name'].append([re.sub('_',' ',i).capitalize() for i in final['name'][i*4:(i+1)*4]])
        tab['category'].append([re.sub('_',' ',i).capitalize() for i in final['category'][i*4:(i+1)*4]])
        tab['location'].append(["("+str(i[0])+","+str(i[1])+")" for i in final['location'][i*4:(i+1)*4]])
        tab['price'].append([str(i) for i in final['price'][i*4:(i+1)*4]])
        tab['rating'].append([str(i) for i in final['rating'][i*4:(i+1)*4]])
        #print('Final  Recommendations are: ',price, rating,location,category,name)
        
    
    return tab




















cat_rating = {'luxury_&_special_occasions': 5.0, 'outdoor_activities': 4.0, 'recommended_experiences': 5.0, 'food,_wine_&_nightlife': 4.0, 'cruises,_sailing_&_water_tours': 3.0, 'cultural_&_theme_tours': 5.0}


def feed_input(price_low,price_high,province,days,cat_rating):
    
    att_df = pd.read_json('./etl/attractions.json',orient='records')
    category_df = att_df.groupby('category').size().reset_index().sort_values([0],ascending=False)[:20]
    categories = list(category_df.category.values)
    filename, user, rbm_att = get_recc(att_df, cat_rating)
    with_url = filter_df(filename, user, price_low, price_high, province, att_df)

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

import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from IPython.display import display
class RBM(object):
    '''
    Class definition for a simple RBM
    '''
    def __init__(self, alpha, H, num_vis):

        self.alpha = alpha
        self.num_hid = H
        self.num_vis = num_vis # might face an error here, call preprocess if you do
        self.errors = []
        self.energy_train = []
        self.energy_valid = []

    def training(self, train, valid, user, epochs, batchsize, free_energy, verbose, filename):
        '''
        Function where RBM training takes place
        '''
        print('inside rbm training')
        vb = tf.placeholder(tf.float32, [self.num_vis]) # Number of unique books
        hb = tf.placeholder(tf.float32, [self.num_hid]) # Number of features were going to learn
        W = tf.placeholder(tf.float32, [self.num_vis, self.num_hid])  # Weight Matrix
        v0 = tf.placeholder(tf.float32, [None, self.num_vis])

        print("Phase 1: Input Processing")
        _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
        # Gibb's Sampling
        h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
        print("Phase 2: Reconstruction")
        _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
        v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
        h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

        print("Creating the gradients")
        w_pos_grad = tf.matmul(tf.transpose(v0), h0)
        w_neg_grad = tf.matmul(tf.transpose(v1), h1)

        # Calculate the Contrastive Divergence to maximize
        CD = (w_pos_grad - w_neg_grad) / tf.cast(tf.shape(v0)[0], tf.float32)

        # Create methods to update the weights and biases
        update_w = W + self.alpha * CD
        update_vb = vb + self.alpha * tf.reduce_mean(v0 - v1, 0)
        update_hb = hb + self.alpha * tf.reduce_mean(h0 - h1, 0)

        # Set the error function, here we use Mean Absolute Error Function
        err = v0 - v1
        err_sum = tf.reduce_mean(err * err)

        # Initialize our Variables with Zeroes using Numpy Library
        # Current weight
        cur_w = np.zeros([self.num_vis, self.num_hid], np.float32)
        # Current visible unit biases
        cur_vb = np.zeros([self.num_vis], np.float32)

        # Current hidden unit biases
        cur_hb = np.zeros([self.num_hid], np.float32)

        # Previous weight
        prv_w = np.random.normal(loc=0, scale=0.01,
                                size=[self.num_vis, self.num_hid])
        # Previous visible unit biases
        prv_vb = np.zeros([self.num_vis], np.float32)

        # Previous hidden unit biases
        prv_hb = np.zeros([self.num_hid], np.float32)

        print("Running the session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        print("Training RBM with {0} epochs and batch size: {1}".format(epochs, batchsize))
        print("Starting the training process")
        util = Util()
        for i in range(epochs):
            for start, end in zip(range(0, len(train), batchsize), range(batchsize, len(train), batchsize)):
                batch = train[start:end]
                cur_w = sess.run(update_w, feed_dict={
                                 v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_vb = sess.run(update_vb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_hb = sess.run(update_hb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                prv_w = cur_w
                prv_vb = cur_vb
                prv_hb = cur_hb

            if valid:
                etrain = np.mean(util.free_energy(train, cur_w, cur_vb, cur_hb))
                self.energy_train.append(etrain)
                evalid = np.mean(util.free_energy(valid, cur_w, cur_vb, cur_hb))
                self.energy_valid.append(evalid)
            self.errors.append(sess.run(err_sum, feed_dict={
                          v0: train, W: cur_w, vb: cur_vb, hb: cur_hb}))
            if verbose:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
            elif i % 10 == 9:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
        if not os.path.exists('rbm_models'):
            os.mkdir('rbm_models')
        filename = 'rbm_models/'+filename
        if not os.path.exists(filename):
            os.mkdir(filename)
        np.save(filename+'/w.npy', prv_w)
        np.save(filename+'/vb.npy', prv_vb)
        np.save(filename+'/hb.npy',prv_hb)
        
        if free_energy:
            print("Exporting free energy plot")
            self.export_free_energy_plot(filename)
        print("Exporting errors vs epochs plot")
        self.export_errors_plot(filename)
        inputUser = [train[user]]
        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
        return rec, prv_w, prv_vb, prv_hb

    def load_predict(self, filename, train, user):
        vb = tf.placeholder(tf.float32, [self.num_vis]) # Number of unique books
        hb = tf.placeholder(tf.float32, [self.num_hid]) # Number of features were going to learn
        W = tf.placeholder(tf.float32, [self.num_vis, self.num_hid])  # Weight Matrix
        v0 = tf.placeholder(tf.float32, [None, self.num_vis])
        
        prv_w = np.load('recommendations/'+filename+'/w.npy')
        prv_vb = np.load('recommendations/'+filename+'/vb.npy')
        prv_hb = np.load('recommendations/'+filename+'/hb.npy')
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        print("Model restored")
        
        inputUser = [train[user]]
        
        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
        
        return rec, prv_w, prv_vb, prv_hb
        
    def calculate_scores(self, ratings, attractions, rec, user):
        '''
        Function to obtain recommendation scores for a user
        using the trained weights
        '''
        # Creating recommendation score for books in our data
        ratings["Recommendation Score"] = rec[0]

        """ Recommend User what books he has not read yet """
        # Find the mock user's user_id from the data
#         cur_user_id = ratings[ratings['user_id']

        # Find all books the mock user has read before
        visited_places = ratings[ratings['user_id'] == user]['attraction_id']
        visited_places

        # converting the pandas series object into a list
        places_id = visited_places.tolist()

        # getting the book names and authors for the books already read by the user
        places_names = []
        places_categories = []
        places_prices = []
        for place in places_id:
            places_names.append(
                attractions[attractions['attraction_id'] == place]['name'].tolist()[0])
            places_categories.append(
                attractions[attractions['attraction_id'] == place]['category'].tolist()[0])
            places_prices.append(
                attractions[attractions['attraction_id'] == place]['price'].tolist()[0])

        # Find all books the mock user has 'not' read before using the to_read data
        unvisited = attractions[~attractions['attraction_id'].isin(places_id)]['attraction_id']
        unvisited_id = unvisited.tolist()
        
        # extract the ratings of all the unread books from ratings dataframe
        unseen_with_score = ratings[ratings['attraction_id'].isin(unvisited_id)]

        # grouping the unread data on book id and taking the mean of the recommendation scores for each book_id
        grouped_unseen = unseen_with_score.groupby('attraction_id', as_index=False)['Recommendation Score'].max()
        display(grouped_unseen.head())
        
        # getting the names and authors of the unread books
        unseen_places_names = []
        unseen_places_categories = []
        unseen_places_prices = []
        unseen_places_scores = []
        for place in grouped_unseen['attraction_id'].tolist():
            unseen_places_names.append(
                attractions[attractions['attraction_id'] == place]['name'].tolist()[0])
            unseen_places_categories.append(
                attractions[attractions['attraction_id'] == place]['category'].tolist()[0])
            unseen_places_prices.append(
                attractions[attractions['attraction_id'] == place]['price'].tolist()[0])
            unseen_places_scores.append(
                grouped_unseen[grouped_unseen['attraction_id'] == place]['Recommendation Score'].tolist()[0])

        # creating a data frame for unread books with their names, authors and recommendation scores
        unseen_places = pd.DataFrame({
            'att_id' : grouped_unseen['attraction_id'].tolist(),
            'att_name': unseen_places_names,
            'att_cat': unseen_places_categories,
            'att_price': unseen_places_prices,
            'score': unseen_places_scores
        })

        # creating a data frame for read books with the names and authors
        seen_places = pd.DataFrame({
            'att_id' : places_id,
            'att_name': places_names,
            'att_cat': places_categories,
            'att_price': places_prices
        })

        return unseen_places, seen_places

    def export(self, unseen, seen, filename, user):
        '''
        Function to export the final result for a user into csv format
        '''
        # sort the result in descending order of the recommendation score
        sorted_result = unseen.sort_values(
            by='score', ascending=False)
        
        x = sorted_result[['score']].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler((0,5))
        x_scaled = min_max_scaler.fit_transform(x)
        
        sorted_result['score'] = x_scaled
        
        # exporting the read and unread books  with scores to csv files

        seen.to_csv(filename+'/user'+user+'_seen.csv')
        sorted_result.to_csv(filename+'/user'+user+'_unseen.csv')
#         print('The attractions visited by the user are:')
#         print(seen)
#         print('The attractions recommended to the user are:')
#         print(sorted_result)

    def export_errors_plot(self, filename):
        plt.plot(self.errors)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.savefig(filename+"/error.png")

    def export_free_energy_plot(self, filename):
        fig, ax = plt.subplots()
        ax.plot(self.energy_train, label='train')
        ax.plot(self.energy_valid, label='valid')
        leg = ax.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Free Energy")
        plt.savefig(filename+"/free_energy.png")
