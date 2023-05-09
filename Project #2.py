#!/usr/bin/env python
# coding: utf-8

# # Project #2 -- What makes a song popular?

# ### For this project, you are going to utilize what you have learned this past week to try and answer the question of what makes a song popular.  To answer this question, you will be using another popular Kaggle database, the song popularity dataset from Kaggle (or at least one of them!):
# 
# https://www.kaggle.com/datasets/yasserh/song-popularity-dataset
# 
# ### I've already downloaded the file, song_data.csv, into your Numpy & Pandas folder.  All the data comes from Spotify, and as you can see from the link above, Spotify uses 13 measures to classify a song:
# <br>
# <li>Duration</li>
# <li>Acousticness</li>
# <li>Danceability</li>
# <li>Energy</li>
# <li>Instrumentalness</li>
# <li>Key</li>
# <li>Liveness</li>
# <li>Audio Mode (Major = 1, Minor = 0...someone musically inclined should explain this to the class)</li>
# <li>Loudness</li>
# <li>Speechiness</li>
# <li>Tempo</li>
# <li>Time Signature (meter signature)</li>
# <li>Audio Valence (how positive/happy/cheerful a song is)</li>
# <br>
#     
# ### Your Task:
# 
# ### Part 1:  Using the tools that you have learned in Pandas (and/or NumPy), do an analysis of the data looking for the factor or combination of factors that most closely predicts the popularity of a song.  Although I am curious to see what each of you comes up with, I'm less interested in your answer than I am in you demonstrating that you can probe a dataset, prepare/analyze the data set, and draw reasonable conclusions from the dataset.  Although you are free to analyze the data however you want, I would like to see examples of all of the following:
# <br>
# <li>Column manipulation (e.g. normalization, binning data, etc.)</li>
# <li>Statistical analysis (e.g. correlation matrix, heat map, etc.)</li>
# <li>Graphing (e.g. bar graphs, scatter plots, histograms</li>
# <br>
# 
# ### Part 2:  Once you have finished your analysis, you will need to produce a slide deck, either with PowerPoint or Google Slides, that makes a case for what factors you believe to be most important.  The deck does not need to be big (5-10 slides) and should not be text heavy.  The deck should, just like any paper, introduce the problem, show your analysis of the data, and end with your conclusions.
# 
# ### Some important notes:
# 
# 1. Just for context, this is the kind of work that you would normally have to do before trying to build a predictive model.  You want to figure out, before building your artificial neural network, what factors most strongly correlate with a particular outcome (or in this case the popularity of the song).  There is no point in using inputs that don't seem to correlate at all to a particular outcome.  That just becomes mathematical clutter, and can negatively impact our ability to make predictions.
# 
# 2. I picked this particular dataset because I think it's a fun one to play with.  We have had a few students work with the SpotiPy (pronounced spot-i-pie, NOT spot-i-pee!!) API to try to build their own song recommendation genus.  So please, have fun with this project.  No need for high-level statistical analyses for those of you in Honors Stats or HCBP&S, unless you find that kind of stuff fun.  
# 
# 3. Unlike previous coding projects, you won't actually write a contiguous program for this one.  PLEASE--use markdown and commenting to walk me through your process.  Use different cells to perform different analyses.  In other words, make it clear and readable.
# 
# 4. You have 4 class periods and 3 nights of homework to complete this project. Give yourself at least one class period and a night of homework to make the slides.  That means you should aim to complete your analysis by Tuesday, May 2nd.

# In[99]:


#importing necessary libraries
import pandas as pd
import seaborn as sn                
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('song_data.csv')
##now, let's get a sense for what the data looks like via df, df.shape, and df.describe()!

# df
# df.nlargest(100, 'song_popularity')
# df.shape
# df.value_counts()
df.describe()## insights: about 18,835 songs, need to normalize data


# In[39]:


##getting a sense for the column distributions by iterating over columns and plotting a histogram!
for column in df.columns:
    plt.hist(df[column], bins = 20) ## changing bins 
    plt.title(column)
    plt.ylabel('Freq') ## titlting y-axis
    plt.show()


# In[40]:


##starting the material for the google slides presentation at this kernel here:
slides_df = pd.read_csv('song_data.csv')
slides_df.nlargest(5, 'song_popularity')
slides_df = slides_df.drop_duplicates(inplace = False)


for column in slides_df.columns:
    if column != 'song_name' and column != 'key' and column!='time_signature' and column!='loudness':#ignore categor.
        pd.to_numeric(column, errors='coerce') ## convert data values from stirng to numerical values (b/c math)
        slides_df[column] = slides_df[column]/(slides_df[column].max())#divide by max,thus normalizing data --> [0,1]
    elif column=='loudness':
        slides_df[column] = slides_df[column]/abs((slides_df[column].max()))#divide by abs value b/c col is log scale

plt.figure(figsize=(16, 8))
sn.set(style="whitegrid")
corr = slides_df.corr()
sn.heatmap(corr,annot=True, cmap="YlGnBu")
#energy and loudness heavily correlated, acousticness and energy anticorrelated;thus, we drop energy from slides_df:


# In[48]:


slides_df.drop('energy', axis=1)
##create filter for highly popular songs (arbitrarily set above 0.7 song_popularity rating)
my_filter = slides_df['song_popularity'] > 0.7
slides_df = slides_df[my_filter]
plt.figure(figsize=(16, 8))
sn.set(style="whitegrid")
corr = slides_df.corr()
sn.heatmap(corr,annot=True, cmap="YlGnBu")
##above is new heatmap for filtered danceability values (>0.7)


# In[47]:


#exploring (via scatterplot) the continous variables that according to the heatmap above, are highly corr w song_pop:
#scatterplot for song danceability vs song_popularity
plt.scatter(slides_df['danceability'],slides_df['song_popularity'])
plt.title('Danceability vs Song Popularity')
plt.xlabel('Song Danceability Scale')
plt.ylabel('Song Popularity Scale')


# In[46]:


#scatterplot for song loudness vs song_popularity
plt.scatter(slides_df['loudness'],slides_df['song_popularity'])
plt.title('Danceability vs Song Loudness')
plt.xlabel('Song Loudness Scale (dB)')
plt.ylabel('Song Popularity Scale')


# In[51]:


#scatterplot for song energy vs song_popularity
plt.scatter(slides_df['energy'],slides_df['song_popularity'])
plt.title('Danceability vs Energy')
plt.xlabel('Song Energy Scale')
plt.ylabel('Song Popularity Scale')


# In[52]:


#scatterplot for song acousticness vs song_popularity
plt.scatter(slides_df['acousticness'],slides_df['song_popularity'])
plt.title('Danceability vs Acousticness')
plt.xlabel('Song Acousticness Scale')
plt.ylabel('Song Popularity Scale')
##in retrospect, I could've looped through the four columns to create scatterplots rather than doing it separately. 


# In[53]:


##We have accessed, via correlation heatmaps and scatterplots, the quant, continous variables. Now, 
# let's access the categorically-natured variables. Let's start by creating a pivot table for key:

#without a loss of generality, we bin song_popularity into low, mid-low, mid-high, and high categories 
#corresponding to song popularities on the intervals [0,0.25), [0.25, 0.50), [0.50, 0.75), and [0.75, 1.00). 
#We then create a pivot table to summarize where these categories fall for various keys. 


# In[136]:


my_df = pd.read_csv('song_data.csv')
my_df = my_df.drop_duplicates()

for column in my_df.columns:
    if column != 'song_name' and column != 'key' and column!='time_signature' and column!='loudness':#ignore categor.
        pd.to_numeric(column, errors='coerce') ## convert data values from stirng to numerical values (b/c math)
        my_df[column] = my_df[column]/(my_df[column].max())#divide by max,thus normalizing data --> [0,1]

# # my_df.shape ## 18835
my_df['song_popularity'] = pd.cut(my_df['song_popularity'], bins=[0, 0.25, 0.50, 0.75, 1.0],
                                  labels=['Low', 'Mid-Low', 'Mid-High', 'High'])

my_df['song_popularity'].value_counts()


# In[144]:



my_pivot_table = pd.pivot_table(my_df, index = ['key'], values = ['song_name'], 
                     columns = ['song_popularity'], aggfunc = "count")

print(my_pivot_table)
total_songs_by_key = my_df['key'].value_counts()
percentages = my_pivot_table.apply(lambda x: x/total_songs_by_key[x.name], axis=1)
# # ## according to Pandas documentation, .name returns 'index or multiIndex name.''

# # # axis parameter in the apply() method is set to 1 to apply the function to each row of the pivot table. 
# # ##using apply to apply a function
# # #the number of songs in each popularity category for the key are being divided by total num songs in that key
# # #to get the percentage of songs in each popularity category for each key.
print(percentages)
# # ## percentages is new pivot table!!

percentages[('High+Mid-High', '')] = percentages[('song_name', 'Mid-High')] + percentages[('song_name', 'High')]
percentages[('Low+Mid-Low', '')] = percentages[('song_name', 'Mid-Low')] + percentages[('song_name', 'Low')]
percentages[('Diff', '')] = percentages[('High+Mid-High', '')]-percentages[('Low+Mid-Low', '')]
percentages


# In[ ]:


##after analyzing songs by key (and noticing some differences in the 'dif' column),let's now analyze 
# the other categorically-natured variables: audio_mode and time_sig! We can do this via handy boxplots!


# In[58]:


print(slides_df.boxplot(column = 'song_popularity', by='audio_mode'))
##via the boxplot, there seems to be no significant difference in song_popularity by audio_mode


# In[59]:


print(slides_df.boxplot(column = 'song_popularity', by='time_signature'))
####via the boxplot, there might be a significant difference in song_popularity by time_signature. Let's explore
##this idea further and perhaps more robustly using an ANOVA statistical test, below:


# In[61]:


df_for_audio_mode = pd.read_csv('song_data.csv').drop_duplicates(inplace = False)


# In[63]:


##syntax for ANOVA test obtained here: https://towardsdatascience.com/anova-test-with-python-cfbf4013328b
## basically, trying to access if there is a statistical difference of mean song popularity based on various time_sigs
##using ANOVA because the data is a sample of all songs on spotify, not a population!!
import random
my_random_sample = df_for_audio_mode['song_name'].sample(n=500)#sample of 500 songs (note: I'd defined df_for_audio_mode previously, but its essentially the same as slides_df)
random_df = pd.DataFrame(my_random_sample)
# random_df

sample_df = df_for_audio_mode[df_for_audio_mode['song_name'].isin(my_random_sample)].reset_index(drop=True)
sample_df ## is random data frame
sample_df = sample_df[['song_popularity', 'time_signature']] ## get song_popularity and time_signature

groups = sample_df.groupby('time_signature').count().reset_index()
print(groups)


# In[64]:


##in order to run an ANOVA test, we must first meet a few conditions. The first is normality of the sampling dist:
##checking normality assumption; The Q-Q plot shows a largely straight-line pattern if it 
#is from a normal distribution. From the below figures, we may assume that the data for each group falls 
#roughly on a straight line. So, our samples are normally distributed, as hoped to satisfy before ANOVA test. 
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt

unique_time_sigs = sample_df['time_signature'].unique()
for time_sig in unique_time_sigs:
    stats.probplot(sample_df[sample_df['time_signature'] == time_sig]['song_popularity'], dist="norm", plot=plt)
    plt.title("Probability Plot; Time Signature = "+str(time_sig))
    plt.show()


# In[65]:


#Homogeneity of variance Assumption Check:
ratio = sample_df.groupby('time_signature').std().max() / sample_df.groupby('time_signature').std().min()
print(ratio)
##The ratio of the largest to the smallest sample standard deviation is about 1.28
#That is less than the threshold of 2. Thus, we conclude that the assumptions to run an ANOVA are fulfilled!


# In[72]:


# According to process of hypothesis testing for an ANOVA:
# H₀: μ₁= μ₂ = μ₃ = μ4 (null; for the four different time_signature (1,3,4, and 5)!
# H₁: Not all song_popularities means are time_signature equal (alternate hypothesis)
# α = 0.05 (significance level)

##initializing empty lists to put in for each different key
empty_list_4 = []
empty_list_3 = []
empty_list_5 = []
empty_list_1 = []
# sample_df['time_signature'].value_counts() 
#time sigs are 4, 3, 5, 1
for index, row in sample_df.iterrows():
    if row[1] == 4: ## ie if time_signautre is 4 (4/4 time)
        empty_list_4.append(row[0]) ## add it to initially empty list that will hold time_signatures of 4
    elif row[1] == 3:
        empty_list_3.append(row[0])
    elif row[1] == 5:
        empty_list_5.append(row[0])
    elif row[1] == 1: ## ie if time_sig is 1/4
        empty_list_1.append(row[0])
        
        

##import necessary library to get f_oneaway -- the method to run an ANOVA test in python!
import numpy as np
from scipy.stats import f_oneway

f_oneway(empty_list_4, empty_list_3, empty_list_5, empty_list_1)
##since p-value (.23) >.05, we cannot reject the null hypothesis. 
#there is not sufficient evidence to say that there is 
#statistically significant difference between the song popularity scores of the various time_signatures groups!

#despite the initially decieving boxplot that hinted that time_sig played a role, our ANOVA test tells us otherwise!!

