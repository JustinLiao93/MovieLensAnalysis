#import package
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#change working dictionary
os.chdir("C:/Users/Justin/Documents/ml-20m")   
os.getcwd()

#read data
movies = pd.read_csv('movies.csv',sep = ',')
tags = pd.read_csv('tags.csv',sep = ',')
ratings = pd.read_csv('ratings.csv',sep = ',')
g_scores = pd.read_csv('genome-scores.csv',sep = ',')
g_tags = pd.read_csv('genome-tags.csv',sep = ',')

#basic info
movies['movieId'].nunique() #movie中有27278部電影
ratings['userId'].nunique() #共有138493個使用者對電影評分
g_scores[['movieId','relevance']].groupby('movieId').size() #每個都是1128tags
g_tags['tagId'].nunique() #共有1128個tags

#movies
movies.dtypes
movies['year'] = movies['title'].str.extract('.*\((.*)\).*', expand=True)
movies = movies.drop(22679, axis = 0)
movies = movies.drop(19859, axis = 0)
movies = movies.drop(15646, axis = 0)
movies = movies.drop(17341, axis = 0)
movies = movies.drop(22368, axis = 0)
movies = movies.drop(22669, axis = 0)
movies['year'] = movies['year'].astype('category')
movies['title'] = movies['title'].str[:-7]

genre_list=['Action','Adventure','Animation','Children','Comedy','Crime','Documentary',
            'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance',
            'Sci-Fi','Thriller','War','Western']
for each_genre in genre_list:
    movies['is' + each_genre] = movies['genres'].str.contains(each_genre)

movies = movies.dropna() #有17部電影沒有年分 27255


data = pd.merge(ratings, movies)
del data['timestamp']; del data['genres']
data['movieId'].nunique() #movie中有26721部電影 有517電影沒有使用者評分過
data['userId'].nunique() #共有138493個使用者對電影評分

print(data['year'].isnull().any().any())

ratings_user_count = ratings[['movieId','userId']].groupby('movieId').count()
ratings_user_count.rename(columns={'userId':'ratings_user_count'}, inplace = True)
ratings_mean = ratings[['movieId','rating']].groupby('movieId').mean()
ratings_mean.rename(columns={'rating':'ratings_mean'}, inplace = True)

data = pd.merge(movies,ratings_user_count, on='movieId')
data = pd.merge(data,ratings_mean, on='movieId')

Y1 = data['ratings_user_count'];Y2 = data['ratings_mean']
X = data.drop(['movieId','title','genres','ratings_user_count','ratings_mean'], axis = 1 )

#切分訓練與驗證樣本
from sklearn.model_selection import train_test_split

x_train, x_test, y1_train, y1_test = train_test_split(X, Y1, test_size = 0.2)
x_train, x_test, y2_train, y2_test = train_test_split(X, Y2, test_size = 0.2)
#linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model1 = lm.fit(x_train,y1_train)
model2 = lm.fit(x_train,y2_train)
predictions = lm.predict(X)
model1.score(X,Y1) #R-square
model2.score(X,Y2) #R-square
model1.coef_
model2.coef_

#label binning
ratig_freq = ratings[['movieId','rating']].groupby('rating').count()

cut_points = [0,2.5,3.5,5]
#ratings['rating_bin'] = pd.cut(ratings['rating'], cut_points, labels = [0,1,2])
#ratings[['movieId','rating_bin']].groupby('rating_bin').count()
data['rating_bin'] = pd.cut(data['ratings_mean'], cut_points, labels = [0,1,2])
#data.dtypes

Y3 = data['rating_bin']
X = data.drop(['movieId','title','genres','ratings_user_count','ratings_mean','rating_bin'], axis = 1 )
x_train, x_test, y3_train, y3_test = train_test_split(X, Y3, test_size = 0.4)

def training():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_selection import f_regression
    from sklearn.metrics import classification_report
    from sklearn.svm import LinearSVC #基於線性假設的支援向量機分類器LinearSVC
    from sklearn.neighbors import KNeighborsClassifier #k鄰近分類器
    from sklearn.tree import DecisionTreeClassifier #決策樹分類器
    from sklearn.ensemble import RandomForestClassifier #隨機森林分類器
    from sklearn.ensemble import GradientBoostingClassifier #梯度提升決策樹
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import accuracy_score
    
    lr = LogisticRegression() #initialize Logistic Regression
    lr.fit(x_train,y3_train) #train
    lr_y_predict = lr.predict(x_test) #predict
    
    lsvc = LinearSVC() #初始化線性假設的支援向量機分類器
    lsvc.fit(x_train,y3_train)
    lsvc_y_predict = lsvc.predict(x_test)
    
    knc = KNeighborsClassifier()
    knc.fit(x_train,y3_train)
    knc_y_predict = knc.predict(x_test)
    
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train,y3_train)
    dtc_y_predict = dtc.predict(x_test)
    
    rfc = RandomForestClassifier()
    rfc.fit(x_train,y3_train)
    rfc_y_pred = rfc.predict(x_test)
    
    gbc = GradientBoostingClassifier()
    gbc.fit(x_train,y3_train)
    gbc_y_pred = gbc.predict(x_test)

    voting_clf = VotingClassifier( estimators=[("lr", lr), ("svm", lsvc), ("knc", knc), ("dtc", dtc)], voting="hard" )

    for clf in ( lr, lsvc, knc, dtc, voting_clf ):
        clf.fit( x_train, y3_train )
        y_pred = clf.predict( x_test )
    #print( clf.__class__.__name__, accuracy_score(y3_test, y_pred) )
    #print( voting_clf.score(x_train,y3_train) )
    #print (classification_report(y3_test,y_pred,target_names = ['not rec','rec', 'very re']))
    
    print ('The Accuracy of EnsembleLearning: ', voting_clf.score(x_train,y3_train))
    print ('The Accuracy of EnsembleLearning: ', voting_clf.score(x_test,y3_test))
    print (classification_report(y_pred, y3_test))
        
    #print ('Accuracy of LR Classifier:', lr.score(x_train,y3_train))
    #print ('Accuracy of LR Classifier:', lr.score(x_test,y3_test))
    #print(f_regression(x_train,y3_train)[1])
    #print (classification_report(y3_test,lr_y_predict,target_names = ['not rec','rec', 'very re']))
    #print ('---------------------------------------------------------')
    #print ('The Accuracy of Linear SVC is',lsvc.score(x_train,y3_train))
    #print ('The Accuracy of Linear SVC is',lsvc.score(x_test,y3_test))
    #print (classification_report(y3_test,lsvc_y_predict,target_names=['not rec','rec', 'very re']))
    #print ('---------------------------------------------------------')
    #print ('The Accuracy of K-nearest Neighbor Classifier is ', knc.score(x_train,y3_train))
    #print ('The Accuracy of K-nearest Neighbor Classifier is ', knc.score(x_test,y3_test))
    #print (classification_report(y3_test,knc_y_predict,target_names=['not rec','rec', 'very re']))
    #print ('---------------------------------------------------------')
    #print ('Decision Tree Score: ', dtc.score(x_train,y3_train))
    #print ('Decision Tree Score: ', dtc.score(x_test,y3_test))
    #print (classification_report(y3_test, dtc_y_predict,target_names=['not rec','rec', 'very re']))
    print ('---------------------------------------------------------')
    print ('The Accuracy of RandomForestClassifier: ', rfc.score(x_train,y3_train))
    print ('The Accuracy of RandomForestClassifier: ', rfc.score(x_test,y3_test))
    print (classification_report(rfc_y_pred, y3_test))
    print ('---------------------------------------------------------')
    print ('The Accuracy of GradientBoostingClassifier: ', gbc.score(x_train,y3_train))
    print ('The Accuracy of GradientBoostingClassifier: ', gbc.score(x_test,y3_test))
    print (classification_report(gbc_y_pred, y3_test))

training()

#tags 

#前10大標籤
tag_counts = tags['tag'].value_counts().head(10)
#每部電影有幾個標籤
movies_tags_count = tags[['movieId','tag']].groupby('movieId').count()
movies_tags_count.rename(columns={'tag':'movies_tags_count'}, inplace = True)
movies_tags_count.max()
movies_tags_count.mean() #23.82個
#每部電影有幾個使用者貼過標籤
tags_user = tags[['movieId','userId']].drop_duplicates()
movies_users_count = tags_user[['movieId','userId']].groupby('movieId').count()
movies_users_count.rename(columns={'userId':'movies_users_count'}, inplace = True)
movies_users_count.max()
movies_users_count.mean() #8.95個

tag_cnt = tags[['movieId','tag','userId']].groupby(['movieId','tag']).count()
tag_cnt.rename(columns={'userId':'tag_cnt'}, inplace = True)

tag_cnt = pd.merge(tags,tag_cnt, on=['movieId','tag'])
tag_cnt.sort_values(by=['movieId','tag_cnt'], inplace=True, ascending=False)
tag_cnt = tag_cnt[['movieId','tag','tag_cnt']].drop_duplicates()
tag_cnt.sort_values(by=['movieId','tag_cnt'], inplace=True, ascending=False)
#每部電影前10大標籤
top_tag_cnts = pd.DataFrame(tag_cnt.groupby(['movieId']).head(10))
#transpose
tag_data = top_tag_cnts.pivot(index='movieId', columns='tag', values='tag_cnt').reset_index().fillna(0)

#feature extraction
data['year_bin'] = data['year'].astype('float')
cut_points_years = [0,1980,2000,2020]
data['year_bin'] = pd.cut(data['year_bin'], cut_points_years, labels = [0,1,2])

#計算各類別 不推薦%
data_2 = pd.merge(data,tag_data, on='movieId')
genres_list_2 = ['isAction','isAdventure','isAnimation','isChildren','isComedy','isCrime',
                    'isDocumentary','isDrama','isFantasy','isFilm-Noir','isHorror','isMusical',
                    'isMystery','isRomance','isSci-Fi','isThriller','isWar','isWestern']
tmp = pd.DataFrame()
for val in range(len(genres_list_2)):
    tmp['filter' + genres_list_2[val]] = (data_2[genres_list_2[val]] == True)
genres_list_3 = ['filterisAction','filterisAdventure','filterisAnimation','filterisChildren','filterisComedy','filterisCrime',
                    'filterisDocumentary',
               'filterisDrama','filterisFantasy','filterisFilm-Noir','filterisHorror','filterisMusical','filterisMystery','filterisRomance',
               'filterisSci-Fi','filterisThriller','filterisWar','filterisWestern']
for val in range(len(genres_list_3)):
    print(round((data_2[tmp[genres_list_3[val]]]['rating_bin'].value_counts()[0])/
      (data_2[tmp[genres_list_3[val]]]['rating_bin'].value_counts().sum()),3))
'''
Horror	    0.288
Sci-Fi	    0.207
Children	0.174
Action	    0.154
Fantasy	    0.131
Comedy	    0.127
Adventure	0.125
Thriller	0.124
Musical	    0.097
Mystery	    0.091
Animation	0.086
Western	    0.081
Crime	    0.073
Romance	    0.063
Drama	    0.058
Documentary	0.056
War	        0.054
Film-Noir	0.016
'''
    

#依據高低，共分成4類
data_2['isGenre1'] = data_2['genres'].str.contains('Drama|Documentary|War|Film-Noir')
data_2['isGenre2'] = data_2['genres'].str.contains('Musical|Mystery|Animation|Western|Crime|Romance')
data_2['isGenre3'] = data_2['genres'].str.contains('Fantasy|Comedy|Adventure|Thriller')
data_2['isGenre4'] = data_2['genres'].str.contains('Horror|Sci-Fi|Children|Action|(no genres listed)|IMAX')

#data_2 = data_2.drop(['isAction','isAdventure','isAnimation','isChildren','isComedy',
#                      'isCrime','isDocumentary','isDrama','isFantasy','isFilm-Noir',
#                      'isHorror','isMusical','isMystery','isRomance','isSci-Fi',
#                      'isThriller','isWar','isWestern'], axis = 1 )

genres_list_4 = ['isGenre1','isGenre2','isGenre3','isGenre4']
for val in range(len(genres_list_4)):
    tmp['filter' + genres_list_4[val]] = (data_2[genres_list_4[val]] == True)
    
tmp_G1 = data_2[tmp['filterisGenre1']].drop(['title','genres','year','ratings_user_count','ratings_mean','year_bin','rating_bin',
               'isGenre2','isGenre3','isGenre4'], axis = 1 ).drop_duplicates()
tmp_G2 = data_2[tmp['filterisGenre2']].drop(['title','genres','year','ratings_user_count','ratings_mean','year_bin','rating_bin',
               'isGenre1','isGenre3','isGenre4'], axis = 1 ).drop_duplicates()
tmp_G3 = data_2[tmp['filterisGenre3']].drop(['title','genres','year','ratings_user_count','ratings_mean','year_bin','rating_bin',
               'isGenre2','isGenre1','isGenre4'], axis = 1 ).drop_duplicates()
tmp_G4 = data_2[tmp['filterisGenre4']].drop(['title','genres','year','ratings_user_count','ratings_mean','year_bin','rating_bin',
               'isGenre2','isGenre3','isGenre1'], axis = 1 ).drop_duplicates()

'''
tmp_G1_tag = pd.DataFrame(tmp_G1.drop(['movieId','isGenre1'], axis = 1).sum().nlargest(5))
tmp_G1_tag['Genre'] = 1; tmp_G1_tag['index'] = tmp_G1_tag.index
tmp_G1_tag = tmp_G1_tag.pivot(index='Genre', columns= 0, values='index').reset_index()
tmp_G1_tag.rename(columns={1284.0:'top1',1288.0:'top2',1473.0:'top3',1473.0:'top1'}, inplace = True)
'''
tmp_G1_tag = pd.DataFrame(tmp_G1.drop(['movieId','isGenre1'], axis = 1).sum().nlargest(5))
tmp_G1_tag['Genre'] = '1'; tmp_G1_tag['index'] = tmp_G1_tag.index
tmp_G1_tag = tmp_G1_tag.pivot(index='Genre', columns='index', values=0).reset_index()

tmp_G2_tag = pd.DataFrame(tmp_G2.drop(['movieId','isGenre2'], axis = 1).sum().nlargest(5))
tmp_G2_tag['Genre'] = '2'; tmp_G2_tag['index'] = tmp_G2_tag.index
tmp_G2_tag = tmp_G2_tag.pivot(index='Genre', columns='index', values=0).reset_index()

tmp_G3_tag = pd.DataFrame(tmp_G3.drop(['movieId','isGenre3'], axis = 1).sum().nlargest(5))
tmp_G3_tag['Genre'] = '3'; tmp_G3_tag['index'] = tmp_G3_tag.index
tmp_G3_tag = tmp_G3_tag.pivot(index='Genre', columns='index', values=0).reset_index()

tmp_G4_tag = pd.DataFrame(tmp_G4.drop(['movieId','isGenre4'], axis = 1).sum().nlargest(5))
tmp_G4_tag['Genre'] = '4'; tmp_G4_tag['index'] = tmp_G4_tag.index
tmp_G4_tag = tmp_G4_tag.pivot(index='Genre', columns='index', values=0).reset_index()

#filter mask

data_2 = data_2[['movieId','title','genres','year','ratings_mean',
                      'year_bin','rating_bin','isGenre1','isGenre2','isGenre3',
                      'isGenre4','isAction','isAdventure','isAnimation','isChildren','isComedy',
                      'isCrime','isDocumentary','isDrama','isFantasy','isFilm-Noir',
                      'isHorror','isMusical','isMystery','isRomance','isSci-Fi',
                      'isThriller','isWar','isWestern']]


mask1 = data_2['isGenre1'] == True;mask2 = data_2['isGenre2'] == True;mask3 = data_2['isGenre3'] == True;mask4 = data_2['isGenre4'] == True
mask5 = data_2['isGenre1'] == False;mask6 = data_2['isGenre2'] == False;mask7 = data_2['isGenre3'] == False;mask8 = data_2['isGenre4'] == False

s1 = data_2[(mask1 & mask2 & mask3 & mask4)].copy();s1.loc[:, 'Genre'] = '1234'
s2 = data_2[(mask1 & mask2 & mask3 & mask8)].copy();s2.loc[:, 'Genre'] = '123'
s3 = data_2[(mask5 & mask2 & mask3 & mask4)].copy();s3.loc[:, 'Genre'] = '234'
s4 = data_2[(mask5 & mask2 & mask7 & mask4)].copy();s4.loc[:, 'Genre'] = '24'
s5 = data_2[(mask5 & mask2 & mask3 & mask8)].copy();s5.loc[:, 'Genre'] = '23'
s6 = data_2[(mask5 & mask6 & mask3 & mask4)].copy();s6.loc[:, 'Genre'] = '34'
s7 = data_2[(mask1 & mask2 & mask7 & mask8)].copy();s7.loc[:, 'Genre'] = '12'
s8 = data_2[(mask1 & mask6 & mask3 & mask8)].copy();s8.loc[:, 'Genre'] = '13'
s9 = data_2[(mask1 & mask6 & mask7 & mask4)].copy();s9.loc[:, 'Genre'] = '14'
s10 = data_2[(mask1 & mask6 & mask7 & mask8)].copy();s10.loc[:, 'Genre'] = '1'
s11 = data_2[(mask5 & mask2 & mask7 & mask8)].copy();s11.loc[:, 'Genre'] = '2'
s12 = data_2[(mask5 & mask6 & mask3 & mask8)].copy();s12.loc[:, 'Genre'] = '3'
s13 = data_2[(mask5 & mask6 & mask7 & mask4)].copy();s13.loc[:, 'Genre'] = '4'
s14 = data_2[(mask1 & mask2 & mask7 & mask4)].copy();s14.loc[:, 'Genre'] = '124'

data_3 = s1.append([s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14], ignore_index=True)

tag_data = pd.concat([tmp_G1_tag, tmp_G2_tag, tmp_G3_tag, tmp_G4_tag], ignore_index=True).fillna(0)
del tag_data['Genre']
filter1 = tag_data.loc[0];filter2 = tag_data.loc[1];filter3 = tag_data.loc[2];filter4 = tag_data.loc[3];

ss1 = pd.DataFrame(filter1+filter2+filter3+filter4).sort_values(by=0, ascending=False).head(5);ss1['Genre'] = '1234';ss1['index'] = ss1.index
ss1 = ss1.pivot(index='Genre', columns='index', values=0).reset_index()
ss2 = pd.DataFrame(filter1+filter2+filter3).sort_values(by=0, ascending=False).head(5);ss2['Genre'] = '123';ss2['index'] = ss2.index
ss2 = ss2.pivot(index='Genre', columns='index', values=0).reset_index()
ss3 = pd.DataFrame(filter2+filter3+filter4).sort_values(by=0, ascending=False).head(5);ss3['Genre'] = '234';ss3['index'] = ss3.index
ss3 = ss3.pivot(index='Genre', columns='index', values=0).reset_index()
ss4 = pd.DataFrame(filter2+filter4).sort_values(by=0, ascending=False).head(5);ss4['Genre'] = '24';ss4['index'] = ss4.index
ss4 = ss4.pivot(index='Genre', columns='index', values=0).reset_index()
ss5 = pd.DataFrame(filter2+filter3).sort_values(by=0, ascending=False).head(5);ss5['Genre'] = '23';ss5['index'] = ss5.index
ss5 = ss5.pivot(index='Genre', columns='index', values=0).reset_index()
ss6 = pd.DataFrame(filter3+filter4).sort_values(by=0, ascending=False).head(5);ss6['Genre'] = '34';ss6['index'] = ss6.index
ss6 = ss6.pivot(index='Genre', columns='index', values=0).reset_index()
ss7 = pd.DataFrame(filter1+filter2).sort_values(by=0, ascending=False).head(5);ss7['Genre'] = '12';ss7['index'] = ss7.index
ss7 = ss7.pivot(index='Genre', columns='index', values=0).reset_index()
ss8 = pd.DataFrame(filter1+filter3).sort_values(by=0, ascending=False).head(5);ss8['Genre'] = '13';ss8['index'] = ss8.index
ss8 = ss8.pivot(index='Genre', columns='index', values=0).reset_index()
ss9 = pd.DataFrame(filter1+filter4).sort_values(by=0, ascending=False).head(5);ss9['Genre'] = '14';ss9['index'] = ss9.index
ss9 = ss9.pivot(index='Genre', columns='index', values=0).reset_index()
ss10 = pd.DataFrame(filter1).sort_values(by=0, ascending=False).head(5);ss10['Genre'] = '1';ss10['index'] = ss10.index
ss10 = ss10.pivot(index='Genre', columns='index', values=0).reset_index()
ss11 = pd.DataFrame(filter2).sort_values(by=1, ascending=False).head(5);ss11['Genre'] = '2';ss11['index'] = ss11.index
ss11 = ss11.pivot(index='Genre', columns='index', values=1).reset_index()
ss12 = pd.DataFrame(filter3).sort_values(by=2, ascending=False).head(5);ss12['Genre'] = '3';ss12['index'] = ss12.index
ss12 = ss12.pivot(index='Genre', columns='index', values=2).reset_index()
ss13 = pd.DataFrame(filter4).sort_values(by=3, ascending=False).head(5);ss13['Genre'] = '4';ss13['index'] = ss13.index
ss13 = ss13.pivot(index='Genre', columns='index', values=3).reset_index()
ss14 = pd.DataFrame(filter1+filter2+filter4).sort_values(by=0, ascending=False).head(5);ss14['Genre'] = '124';ss14['index'] = ss14.index
ss14 = ss14.pivot(index='Genre', columns='index', values=0).reset_index()

tag_data_2 = ss1.append([ss2,ss3,ss4,ss5,ss6,ss7,ss8,ss9,ss10,ss11,ss12,ss13,ss14], ignore_index=True).fillna(0)


genre_tag_table = tag_data_2.copy()
del genre_tag_table['Genre']
genre_tag_table[genre_tag_table > 0] = 1
genre_tag_table = pd.concat([tag_data_2['Genre'],genre_tag_table], axis=1 ).astype('category')


data_4 = pd.merge(data_3,genre_tag_table, on = 'Genre')
    
Y3 = data_4['rating_bin']
X = data_4.drop(['movieId','genres','title','year','ratings_mean','rating_bin','Genre'], axis = 1 )
x_train, x_test, y3_train, y3_test = train_test_split(X, Y3, test_size = 0.4)
    
training()    



#plot

movie_summary = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
movie_summary['num_ratings'] = ratings.groupby('movieId')['rating'].count()
plt.figure(figsize=(15,4));
movie_summary['rating'].hist(bins=70)
plt.figure(figsize=(15,4))
movie_summary[movie_summary['num_ratings'] < 200]['num_ratings'].hist(bins=200)


user_summary = pd.DataFrame(ratings.groupby('userId')['rating'].count())
user_summary['rating_mean'] = ratings.groupby('userId')['rating'].mean()
plt.figure(figsize=(15,4))
user_summary['rating_mean'].hist(bins=70)