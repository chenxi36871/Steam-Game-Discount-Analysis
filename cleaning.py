import pandas as pd
from datetime import datetime
import numpy as np
from ast import literal_eval


df_read_full = pd.read_csv('D:/pythonProject/raw_data_full.csv', index_col=0)
before2016 = 0
namenull = 0
for num in range(0, len(df_read_full)):
    try:
        year = int(df_read_full.iloc[num, 17].split('-')[0])
        month = int(df_read_full.iloc[num, 17].split('-')[1])
        date = int(df_read_full.iloc[num, 17].split('-')[2])
        if datetime(year, month, date) < datetime(2016, 1, 1):
            before2016 = before2016 + 1
    except:
        a = 1
    # if pd.isnull(np.datetime64(str(df_read_full.iloc[num, 17]))):
    if str(df_read_full.iloc[num, 17]) == 'nan':
        namenull = namenull+1

# 删除了所有gamename为空的和2016年之前的数据条
df_read_clear = pd.read_csv('D:/pythonProject/raw_data_clear1.csv', index_col=0).drop(['mark'], axis=1)
df_history_price = pd.read_csv('D:/pythonProject/cleaning/history_price_clean.csv', index_col=0, encoding='utf-8')
# df_history_full_price = pd.read_csv('D:/pythonProject/cleaning/full_price_clean.csv', index_col=0, encoding='utf-8')
df_release_price = pd.read_csv('D:/pythonProject/cleaning/release_price_clean.csv', index_col=0, encoding='utf-8')
first_discount = pd.read_csv('D:/pythonProject/cleaning/first_discount_date.csv', index_col=0, encoding='utf-8')
fifty_discount = pd.read_csv('D:/pythonProject/cleaning/fifty_discount_date.csv', index_col=0, encoding='utf-8')
df_steamspy = pd.read_csv('D:/pythonProject/cleaning/steamspy_data.csv', index_col=0, encoding='utf-8')

# 整合所有数据
df_read_clear = df_read_clear.rename(columns={'Unnamed: 0.1': 'index'})
df_release_price = df_release_price.rename(columns={'gameid': 'index'})
first_discount = first_discount.rename(columns={'gameid': 'index'})
fifty_discount = fifty_discount.rename(columns={'gameid': 'index'})
test = pd.merge(df_read_clear, df_release_price, how='left', on='index')
test = pd.merge(test, first_discount, how='left', on='index')
test = pd.merge(test, fifty_discount, how='left', on='index')
test = pd.merge(test, df_steamspy, how='left', on='index')
test.to_csv('D:/pythonProject/cleaning/full_data_raw.csv', encoding='utf_8_sig')




# 加入steamspy的数据
df_steamspy = pd.DataFrame()
for filenum in range(100, 44169, 100): #44169
    path = 'D:/pythonProject/steam_spy/raw_data_steamspy_' + str(filenum) + '.csv'
    df_temp = pd.read_csv(path, index_col=0, encoding='utf-8')
    df_steamspy = pd.concat([df_steamspy, df_temp], axis=0)

df_temp = pd.read_csv('D:/pythonProject/steam_spy/raw_data_steamspy_44169.csv', index_col=0, encoding='utf-8')
df_steamspy = pd.concat([df_steamspy, df_temp], axis=0)
df_steamspy.to_csv('D:/pythonProject/cleaning/steamspy_data.csv', encoding='utf-8')

# 加入价格相关的column
# 发售价格，第一次打折时间距发售时间的天数，打折50%的时间距发售时间的天数
# 读取文件&合并
def readandconcat(filename, axis):
    x1 = pd.read_csv('D:/pythonProject/history_price/' + filename +'_0-1000.csv',
                                    index_col=0, encoding='utf_8_sig')
    x2 = pd.read_csv('D:/pythonProject/history_price/' + filename +'_1000-5000.csv',
                                    index_col=0, encoding='utf_8_sig')
    x3 = pd.read_csv('D:/pythonProject/history_price/' + filename +'_5000-10000.csv',
                                    index_col=0, encoding='utf_8_sig')
    x4 = pd.read_csv('D:/pythonProject/history_price/' + filename +'_10000-15000.csv',
                                    index_col=0, encoding='utf_8_sig')
    x5 = pd.read_csv('D:/pythonProject/history_price/' + filename +'_15000-25000.csv',
                                    index_col=0, encoding='utf_8_sig')
    x6 = pd.read_csv('D:/pythonProject/history_price/' + filename +'_25000-35000.csv',
                                    index_col=0, encoding='utf_8_sig')
    x7 = pd.read_csv('D:/pythonProject/history_price/' + filename +'_35000~40000.csv',
                                    index_col=0, encoding='utf_8_sig')
    x8 = pd.read_csv('D:/pythonProject/history_price/' + filename +'_40000~44170.csv',
                                    index_col=0, encoding='utf_8_sig')
    x = pd.concat([x1, x2, x3, x4, x5, x6, x7, x8], axis=axis)
    # x.to_csv('D:/pythonProject/cleaning/' + filename + '.csv', encoding='utf-8')
    col_null = x.isnull().sum(axis=0)
    x_temp = x.iloc[:5000 - col_null.min(), :]
    x_temp.isnull().sum(axis=axis)
    # x_temp.to_csv('D:/pythonProject/cleaning/'+filename+'.csv', encoding='utf-8')
    return x_temp

history_date = readandconcat('history_date', 1)
history_date = pd.read_csv('D:/pythonProject/cleaning/history_date.csv', index_col=0, encoding='utf_8_sig')
# error_list
error_list = readandconcat('error_list', 0)

# history_price & history_full_price

# df_full_price1 = pd.read_csv('D:/pythonProject/history_price/full_price_0-1000.csv',
#                                index_col=0, encoding='utf_8_sig')
# df_full_price2 = pd.read_csv('D:/pythonProject/history_price/full_price_1000-5000.csv',
#                                index_col=0, encoding='utf_8_sig')
# df_full_price3 = pd.read_csv('D:/pythonProject/history_price/full_price_5000-10000.csv',
#                                index_col=0, encoding='utf_8_sig')
# df_full_price4 = pd.read_csv('D:/pythonProject/history_price/full_price_10000-15000.csv',
#                                index_col=0, encoding='utf_8_sig')
# df_full_price5 = pd.read_csv('D:/pythonProject/history_price/full_price_15000-25000.csv',
#                                index_col=0, encoding='utf_8_sig')
# df_full_price6 = pd.read_csv('D:/pythonProject/history_price/full_price_25000-35000.csv',
#                                index_col=0, encoding='utf_8_sig')
# df_full_price7 = pd.read_csv('D:/pythonProject/history_price/full_price_35000~40000.csv',
#                                index_col=0, encoding='utf_8_sig')
# df_full_price8 = pd.read_csv('D:/pythonProject/history_price/full_price_40000~44170.csv',
#                                index_col=0, encoding='utf_8_sig')
# df_full_price = pd.concat([df_full_price1, df_full_price2, df_full_price3,
#                               df_full_price4, df_full_price5, df_full_price6,
#                               df_full_price7, df_full_price8], axis=1)
# df_full_price.to_csv('D:/pythonProject/cleaning/full_price.csv', encoding='utf-8')
#
# col_null = df_full_price.isnull().sum(axis=0)
# df_full_price_temp = df_full_price.iloc[:5000-col_null.min(), :]
# df_full_price_temp.isnull().sum(axis=1)
# df_full_price_temp.to_csv('D:/pythonProject/cleaning/full_price_small.csv', encoding='utf-8')

# release price (gameid & release_price)
release_price = pd.DataFrame(columns=('gameid', 'release_price'))
gameid_list = list(df_history_full_price)

for x in range(0, df_history_full_price.shape[1]):
    null_num = df_history_full_price.iloc[:, x].isnull().sum()
    gameid = gameid_list[x]
    r_price = df_history_full_price.iloc[len(df_history_full_price) - null_num - 2, x]
    release_price = release_price.append([{'gameid': gameid, 'release_price': r_price}], ignore_index=True)
release_price.to_csv('D:/pythonProject/cleaning/release_price_clean.csv', encoding='utf-8')
df_release_price = pd.read_csv('D:/pythonProject/cleaning/release_price_clean.csv', index_col=0, encoding='utf_8_sig')

# 第一次打折的时间
first_discount = pd.DataFrame(columns=('gameid', 'first_discount_date'))
gameid_list = list(df_history_price)
for x in range(0, df_history_price.shape[1]): # df_history_price.shape[1]
    real_num = len(df_history_price) - df_history_price.iloc[:, x].isnull().sum() - 1
    r_price = df_release_price.iloc[x, 1]
    while real_num >= 0:
        if df_history_price.iloc[real_num, x] < r_price:
            first_discount_date = history_date.iloc[real_num, x]
            first_discount = first_discount.append([{'gameid': gameid_list[x],
                                                     'first_discount_date': first_discount_date}],
                                                   ignore_index=True)
            break
        if real_num == 0:
            first_discount = first_discount.append([{'gameid': gameid_list[x],
                                                     'first_discount_date': 'None'}],
                                                   ignore_index=True)
        real_num = real_num - 1

df_release_price.isnull().sum(axis=0)
# 有4206个game没有价格信息，剩余35620个。None表示没有打折过
first_discount.to_csv('D:/pythonProject/cleaning/first_discount_date.csv', encoding='utf-8')

# 打折50%的时间
fifty_discount = pd.DataFrame(columns=('gameid', 'fifty_discount_date'))
gameid_list = list(df_history_price)
for x in range(0, df_history_price.shape[1]): # df_history_price.shape[1]
    real_num = len(df_history_price) - df_history_price.iloc[:, x].isnull().sum() - 1
    r_price = df_release_price.iloc[x, 1] * 0.5
    while real_num >= 0:
        if df_history_price.iloc[real_num, x] < r_price:
            fifty_discount_date = history_date.iloc[real_num, x]
            fifty_discount = fifty_discount.append([{'gameid': gameid_list[x],
                                                     'fifty_discount_date': fifty_discount_date}],
                                                   ignore_index=True)
            break
        if real_num == 0:
            fifty_discount = fifty_discount.append([{'gameid': gameid_list[x],
                                                     'fifty_discount_date': 'None'}],
                                                   ignore_index=True)
        real_num = real_num - 1
fifty_discount.to_csv('D:/pythonProject/cleaning/fifty_discount_date.csv', encoding='utf-8')

################################################################################################################################
import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt
from ast import literal_eval
from datetime import datetime
# 基本信息：gamename, gameid, gameprice, description, format_release_date
# 基本信息中的list：taglist, developer, language, system, genre, features
# 用户评价：reviewsummary, reviewsummary_forever, score_30days, rating_sample_num_30days,
#         score_forever, rating_sample_num_forever, positive, negative
# 价格:
# 用户信息：owners(区间), average_forever, average_2weeks, median_forever, median_2weeks, ccu
df = pd.read_csv('D:/pythonProject/cleaning/full_data_raw.csv', index_col=0, encoding='utf_8_sig')
df = df.drop(['index', 'gameprice', 'release_date', 'appid'], axis=1)

# 看每个column的取值范围
def union_column_list(df, col_name):
    test = df[col_name].tolist()
    if isinstance(test[0], list):
        for num in range(0, len(test)):
            try:
                test[num] = literal_eval(test[num])
            except:
                test[num] = [test[num]]
    if isinstance(test[0], str):
        for num in range(0, len(test)):
            test[num] = test[num].split(',')
    x = test[0]
    for num in range(1, len(test)):
        y = list(set(x).union(set(test[num])))
        x = y
    return y

# taglist_tuple -- 427种tag
test = df_raw_data['taglist'].tolist()
for num in range(0, len(test)):
        test[num] = literal_eval(test[num])
taglist_tuple = test[0]
for num in range(1, len(test)):
    y = list(set(taglist_tuple).union(set(test[num])))
    taglist_tuple = y
df_taglist_tuple = pd.DataFrame(taglist_tuple)
# df_taglist_tuple.to_csv('D:/pythonProject/cleaning/taglist_tuple.csv', encoding='utf_8_sig')
df_taglist_tuple = pd.read_csv('D:/pythonProject/cleaning/taglist_tuple.csv', index_col=0, encoding='utf_8_sig')

# reviewsummary
# ['特别好评', '多半好评', 'None', '褒贬不一', '好评如潮', '多半差评', nan, '好评']
reviewsummary_tuple = df_raw_data['reviewsummary'].unique()

# reviewsummary_forever
# ['特别好评', '多半好评', '多半差评', '差评', nan, '特别差评', '差评如潮',
#       '1 篇用户评测', '2 篇用户评测', '褒贬不一', '好评', '9 篇用户评测',
#        '好评如潮', '3 篇用户评测', '8 篇用户评测', '7 篇用户评测', '6 篇用户评测',
#        '5 篇用户评测',  '4 篇用户评测']
reviewsummary_forever_tuple = df_raw_data['reviewsummary_forever'].unique()
reviewsummary_f_new =

# language -- 29种语言
test = df_raw_data['language'].tolist()
for num in range(0, len(test)):
        test[num] = literal_eval(test[num])
language_tuple = test[0]
for num in range(1, len(test)):
    y = list(set(language_tuple).union(set(test[num])))
    language_tuple = y
df_language_tuple = pd.DataFrame(taglist_tuple)
# df_language_tuple.to_csv('D:/pythonProject/cleaning/language_tuple.csv', encoding='utf_8_sig')
df_language_tuple = pd.read_csv('D:/pythonProject/cleaning/language_tuple.csv', index_col=0, encoding='utf_8_sig')

# system -- win, mac, linux

# genre -- 51
test = df_raw_data['genre'].tolist()
for num in range(0, len(test)):
    try:
        test[num] = test[num].split(',')
    except:
        test[num] = ''
genre_tuple = test[0]
for num in range(1, len(test)):
    y = list(set(genre_tuple).union(set(test[num])))
    genre_tuple = y
df_genre_tuple = pd.DataFrame(genre_tuple)
# df_genre_tuple.to_csv('D:/pythonProject/cleaning/genre_tuple.csv', encoding='utf_8_sig')
df_genre_tuple = pd.read_csv('D:/pythonProject/cleaning/genre_tuple.csv', index_col=0, encoding='utf_8_sig')

# features -- 43种features
test = df_raw_data['features'].tolist()
for num in range(0, len(test)):
        test[num] = literal_eval(test[num])
features_tuple = test[0]
for num in range(1, len(test)):
    y = list(set(features_tuple).union(set(test[num])))
    features_tuple = y

# owner -- 12种
# ['100,000 .. 200,000',
#  '20,000,000 .. 50,000,000',
#  '1,000,000 .. 2,000,000',
#  '5,000,000 .. 10,000,000',
#  '2,000,000 .. 5,000,000',
#  '20,000 .. 50,000',
#  '500,000 .. 1,000,000',
#  '200,000 .. 500,000',
#  '0 .. 20,000',
#  '50,000,000 .. 100,000,000',
#  '10,000,000 .. 20,000,000',
#  '50,000 .. 100,000']
test = df_raw_data['owner'].tolist()
for num in range(0, len(test)):
    try:
        test[num] = test[num].split('+')
    except:
        test[num] = ''
owner_tuple = test[0]
for num in range(1, len(test)):
    y = list(set(owner_tuple).union(set(test[num])))
    owner_tuple = y

# steam年度促销节



# developer_tuple = union_column_list('developer')
language_tuple = union_column_list('language')
system_tuple = union_column_list('system')
genre_tuple = union_column_list('genre')
features_tuple = union_column_list('features')

# process missing value
msno.matrix(df_raw_data, labels=True)
plt.show()
msno.bar(df_raw_data)
plt.show()
#  missingno相关性热图措施无效的相关性：一个变量的存在或不存在如何强烈影响的另一个的存在：
msno.heatmap(df_raw_data)
plt.show()


# 字符串'yyyy-mm-dd'转datetime格式
def str_to_date(text):
    temp = text.split('-')
    year = int(temp[0])
    month = int(temp[1])
    day = int(temp[2])
    d = datetime(year, month, day)
    return d



# 处理IsThereAnyDeal里错误数据，删除release date之前的所有数据
history_date = pd.read_csv('D:/pythonProject/cleaning/history_date.csv', index_col=0, encoding='utf_8_sig')
df_history_price = pd.read_csv('D:/pythonProject/cleaning/history_price_small.csv', index_col=0, encoding='utf-8')
df_history_full_price = pd.read_csv('D:/pythonProject/cleaning/full_price_small.csv', index_col=0, encoding='utf-8')


for i in range(0, len(df_raw_data)):
    release_date = str_to_date(df_raw_data.iloc[i, 18])
    index = str(df_raw_data.iloc[i, 0])
    try:
        j = len(history_date.loc[:, index]) - history_date.loc[:, index].isnull().sum() - 1
        while j >= 0:
            if str_to_date(history_date.loc[j, index]) < release_date:
                history_date.loc[j, index] = None
                df_history_price.loc[j, index] = None
            j = j - 1
    except:
        a = 1
df_history_price.to_csv('D:/pythonProject/cleaning/history_price_clean.csv', encoding='utf-8')
history_date.to_csv('D:/pythonProject/cleaning/history_date_clean.csv', encoding='utf-8')

# 获取release price
for i in range(0, history_date.shape[1]):
    length = 2 * (len(history_date.iloc[:, i]) - history_date.iloc[:, i].isnull().sum())
    j = len(df_history_full_price.iloc[:, i]) - df_history_full_price.iloc[:, i].isnull().sum() - 1
    while j >= 0:
        if j + 1 > length:
            df_history_full_price.iloc[j, i] = None
        j = j - 1
df_history_full_price.to_csv('D:/pythonProject/cleaning/full_price_clean.csv', encoding='utf-8')


###########################################################################
# df_raw_data = pd.read_csv('D:/pythonProject/cleaning/full_data_raw.csv', index_col=0, encoding='utf_8_sig')
# df_raw_data_processed = df_raw_data_processed.drop(['index', 'gameprice', 'release_date', 'appid'], axis=1)
# df_raw_data_processed = df_raw_data_processed.dropna(subset=['release_price'])
# df_raw_data_processed.to_csv('D:/pythonProject/cleaning/full_data_notnull.csv', encoding='utf-8')
df_raw_data_processed = pd.read_csv('D:/pythonProject/cleaning/full_data_notnull.csv', index_col=0, encoding='utf_8_sig')
# df_raw_data_processed = df_raw_data_processed.reset_index(drop=True)

df_raw_data_processed['first_discount_period'] = first_discount_period
df_raw_data_processed['fifty_discount_period'] = fifty_discount_period
df_raw_data_processed['reviewsummary_score'] = reviewsummary_score
df_raw_data_processed['reviewsummary_forever_score'] = reviewsummary_forever_score
df_raw_data_processed = df_raw_data_processed.drop(columns=['score_30days'])
df_raw_data_processed['score_30days'] = score_30days
df_raw_data_processed = df_raw_data_processed.drop(columns=['score_forever'])
df_raw_data_processed['score_forever'] = score_forever
df_raw_data_processed = df_raw_data_processed.drop(columns=['rating_sample_num_30days'])
df_raw_data_processed['rating_sample_num_30days'] = rating_sample_num_30days
df_raw_data_processed = df_raw_data_processed.drop(columns=['rating_sample_num_forever'])
df_raw_data_processed['rating_sample_num_forever'] = rating_sample_num_forever
test1 = pd.concat([df_raw_data_processed, system], axis=1)
test1['language_cnt'] = language_count
test1['language_en'] = language_english
test1['language_cn'] = language_chinese
test1['language_ru'] = language_ru
test1['language_spanish'] = language_spanish
test1['language_french'] = language_french
# 调整df_genre列名
a = list(df_genre)
for i in range(0, len(list(df_genre))):
    df_genre = df_genre.rename(columns={a[i]: 'genre_'+a[i]})
test1 = pd.concat([test1, df_genre], axis=1)
# 调整df_features列名
a = list(df_features)
for i in range(0, len(list(df_features))):
    df_features = df_features.rename(columns={a[i]: 'features_'+a[i]})
test1 = pd.concat([test1, df_features], axis=1)
#调整taglist列名
a = list(df_taglist)
for i in range(0, len(list(df_taglist))):
    df_taglist = df_taglist.rename(columns={a[i]: 'tag_'+a[i]})

test1.to_csv('D:/pythonProject/cleaning/full_data_raw0331.csv', encoding='utf_8_sig')

# first_discount_price date to release date -- first_discount_period
# 'None' 表示没有打折过
first_discount_date = df_raw_data_processed['first_discount_date']
release_date = df_raw_data_processed['format_release_date']
first_discount_period = []
for num in range(0, len(df_raw_data_processed)):
    if first_discount_date[num] == 'None':
        first_discount_period.append('None')
        continue
    x = first_discount_date[num].split('-')
    y = release_date[num].split('-')
    a = datetime(int(x[0]), int(x[1]), int(x[2])) - datetime(int(y[0]), int(y[1]), int(y[2]))
    a = a.days
    first_discount_period.append(a)
# first_discount_period = pd.DataFrame(first_discount_period, columns={'period'})
# len(first_discount_period[first_discount_period['period'] == 'None'])

# fifty discount period
fifty_discount_date = df_raw_data_processed['fifty_discount_date']
release_date = df_raw_data_processed['format_release_date']
fifty_discount_period = []
for num in range(0, len(df_raw_data_processed)):
    if fifty_discount_date[num] == 'None':
        fifty_discount_period.append('None')
        continue
    x = fifty_discount_date[num].split('-')
    y = release_date[num].split('-')
    a = datetime(int(x[0]), int(x[1]), int(x[2])) - datetime(int(y[0]), int(y[1]), int(y[2]))
    a = a.days
    fifty_discount_period.append(a)

# reviewsummary
# 9好评如潮，8特别好评，7多半好评，6好评，5褒贬不一，4差评，3多半差评，2特别差评，1差评如潮, 'None', nan
reviewsummary_tuple = df_raw_data_processed['reviewsummary'].unique()
summary_dict = {
    '好评如潮': 9,
    '特别好评': 8,
    '多半好评': 7,
    '好评': 6,
    '褒贬不一': 5,
    '差评': 4,
    '多半差评': 3,
    '特别差评': 2,
    '差评如潮': 1
}

def getsummary(season):
    return summary_dict.get(season, 0)

x = df_raw_data_processed['reviewsummary']
reviewsummary_score = []
for num in range(0, len(df_raw_data_processed)):
    reviewsummary_score.append(getsummary(x[num]))

# reviewsummary
# 9好评如潮，8特别好评，7多半好评，6好评，5褒贬不一，4差评，3多半差评，2特别差评，1差评如潮
# n篇用户评测， nan
df_raw_data_processed['reviewsummary_forever'].unique()
x = df_raw_data_processed['reviewsummary_forever']
reviewsummary_forever_score = []
for num in range(0, len(df_raw_data_processed)):
    reviewsummary_forever_score.append(getsummary(x[num]))

# score 30 days
null_dict ={
    # 'None': np.nan,
    np.nan: 0
}
def getnull(season):
    return null_dict.get(season, season)

x = df_raw_data['score_30days']
for num in range(0, len(df_raw_data)):
    if math.isnan(df_raw_data.loc[num, 'score_30days']):
        df_raw_data.loc[num, 'score_30days'] = 0
    # df_raw_data.loc[num, 'score_30days'] = getnull(x[num])
    # score_30days.append(getnull(x[num]))

# score forever
import math
x = df_raw_data['score_forever']
# score_forever = []
for num in range(0, len(df_raw_data)):
    if math.isnan(df_raw_data.loc[num, 'score_forever']):
        df_raw_data.loc[num, 'score_forever'] = 0
    # df_raw_data_processed.loc[num, 'score_forever'] = getnull(x[num])
    # score_forever.append(getnull(x[num]))

# rating sample num 30days
x = df_raw_data['rating_sample_num_30days']
for num in range(0, len(df_raw_data)):
    if math.isnan(df_raw_data.loc[num, 'rating_sample_num_30days']):
        df_raw_data.loc[num, 'rating_sample_num_30days'] = 0
    # df_raw_data_processed.loc[num, 'rating_sample_num_30days'] = getnull(x[num])
    # rating_sample_num_30days.append(getnull(x[num]))

# rating sample num forever
x = df_raw_data['rating_sample_num_forever']
for num in range(0, len(df_raw_data)):
    if math.isnan(df_raw_data.loc[num, 'rating_sample_num_forever']):
        df_raw_data.loc[num, 'rating_sample_num_forever'] = 0
    # df_raw_data_processed.loc[num, 'rating_sample_num_forever'] = getnull(x[num])
    # rating_sample_num_forever.append(getnull(x[num]))

# system
df_raw_data_processed['system'].unique()
x = df_raw_data_processed['system']

system = pd.DataFrame(columns={'win', 'mac', 'linux'})

for num in range(0, len(df_raw_data_processed)):
    if x[num] == 'win':
        system = system.append([{'win': 1, 'mac': 0, 'linux': 0}], ignore_index=True)
    else:
        if 'win' in literal_eval(x[num]):
            if 'mac' in literal_eval(x[num]):
                if 'linux' in literal_eval(x[num]):
                    system = system.append([{'win': 1, 'mac': 1, 'linux': 1}], ignore_index=True)
                else:
                    system = system.append([{'win': 1, 'mac': 1, 'linux': 0}], ignore_index=True)
            else:
                if 'linux' in literal_eval(x[num]):
                    system = system.append([{'win': 1, 'mac': 0, 'linux': 1}], ignore_index=True)
                else:
                    system = system.append([{'win': 1, 'mac': 0, 'linux': 0}], ignore_index=True)
        else:
            if 'mac' in literal_eval(x[num]):
                if 'linux' in literal_eval(x[num]):
                    system = system.append([{'win': 0, 'mac': 1, 'linux': 1}], ignore_index=True)
                else:
                    system = system.append([{'win': 0, 'mac': 1, 'linux': 0}], ignore_index=True)
            else:
                if 'linux' in literal_eval(x[num]):
                    system = system.append([{'win': 0, 'mac': 0, 'linux': 1}], ignore_index=True)
                else:
                    system = system.append([{'win': 0, 'mac': 0, 'linux': 0}], ignore_index=True)

# language
# 英语，法语，西班牙语 - 拉丁美洲，俄语，汉语
# 支持语言种类 -- language count
df_raw_data_processed['language'].unique()
x = df_raw_data_processed['language']

language_count = []
for num in range(0, len(df_raw_data_processed)):
    language_count.append(len(literal_eval(x[num])))

language_english = []
for num in range(0, len(df_raw_data_processed)):
    if '英语' in literal_eval(x[num]):
        language_english.append(1)
    else:
        language_english.append(0)

language_french = []
for num in range(0, len(df_raw_data_processed)):
    if '法语' in literal_eval(x[num]):
        language_french.append(1)
    else:
        language_french.append(0)

language_spanish = []
for num in range(0, len(df_raw_data_processed)):
    if '西班牙语 - 拉丁美洲' in literal_eval(x[num]):
        language_spanish.append(1)
    else:
        language_spanish.append(0)

language_chinese = []
for num in range(0, len(df_raw_data_processed)):
    if '简体中文' in literal_eval(x[num]) or '繁体中文' in literal_eval(x[num]) :
        language_chinese.append(1)
    else:
        language_chinese.append(0)

language_ru= []
for num in range(0, len(df_raw_data_processed)):
    if '俄语' in literal_eval(x[num]):
        language_ru.append(1)
    else:
        language_ru.append(0)

# test1.to_csv('D:/pythonProject/cleaning/full_data_clean_0330.csv', encoding='utf_8_sig')

# genre
test = df_raw_data_processed['genre'].tolist()
for num in range(0, len(test)):
    try:
        test[num] = test[num].split(',')
    except:
        test[num] = ''
# 去掉list里字符串里的空格
for num in range(0, len(test)):
    test[num] = [x.strip() for x in test[num] if x.strip() != '']
# 获取tuple
genre_tuple = test[0]
for num in range(1, len(test)):
    y = list(set(genre_tuple).union(set(test[num])))
    genre_tuple = y

genre_full = []
for num in range(0, len(test)):
    if test[num] == '':
        continue
    genre_full = genre_full + test[num]
# 统计genre词频
from collections import Counter
genre_freq = pd.DataFrame([Counter(genre_full)]).T
# 转换变量
df_genre = pd.DataFrame(columns=genre_tuple)
for num in range(0, len(test)):
    for i in genre_tuple:
        if i in test[num]:
            df_genre.loc[num, i] = 1
        else:
            df_genre.loc[num, i] = 0
df_genre.to_csv('D:/pythonProject/cleaning/var_genre.csv', encoding='utf_8_sig')


# features
test = df_raw_data_processed['features'].tolist()
for num in range(0, len(test)):
        test[num] = literal_eval(test[num])
# 获取tuple
features_tuple = test[0]
for num in range(1, len(test)):
    y = list(set(features_tuple).union(set(test[num])))
    features_tuple = y

features_full = []
for num in range(0, len(test)):
    if test[num] == '':
        continue
    features_full = features_full + test[num]
# 统计词频
from collections import Counter
features_freq = pd.DataFrame([Counter(features_full)]).T
# 转换变量
df_features = pd.DataFrame(columns=features_tuple)
for num in range(0, len(test)):
    for i in features_tuple:
        if i in test[num]:
            df_features.loc[num, i] = 1
        else:
            df_features.loc[num, i] = 0
df_features.to_csv('D:/pythonProject/cleaning/var_features.csv', encoding='utf_8_sig')



# taglist
test = df_raw_data_processed['taglist'].tolist()
for num in range(0, len(test)):
        test[num] = literal_eval(test[num])
# 获取tuple
taglist_tuple = test[0]
for num in range(1, len(test)):
    y = list(set(taglist_tuple).union(set(test[num])))
    taglist_tuple = y

taglist_full = []
for num in range(0, len(test)):
    if test[num] == '':
        continue
    taglist_full = taglist_full + test[num]
# 统计词频
from collections import Counter
taglist_freq = pd.DataFrame([Counter(taglist_full)]).T
taglist_freq.to_csv('D:/pythonProject/cleaning/freq_taglist.csv', encoding='utf_8_sig')
# 转换变量
df_taglist = pd.DataFrame(columns=taglist_tuple)
for num in range(0, len(test)):
    for i in taglist_tuple:
        if i in test[num]:
            df_taglist.loc[num, i] = 1
        else:
            df_taglist.loc[num, i] = 0
df_taglist.to_csv('D:/pythonProject/cleaning/var_taglist.csv', encoding='utf_8_sig')


# 20210331
# taglist, features, genre降维 -- 取了前7个tag
# y: first_month_discount
# taglist
df_taglist = pd.read_csv('D:/pythonProject/cleaning/var_taglist.csv', index_col=0, encoding='utf-8')
taglist_freq = pd.read_csv('D:/pythonProject/cleaning/freq_taglist.csv', index_col=0, encoding='utf-8')
a = []
for num in range(0, len(taglist_freq)):
    a.append(round(taglist_freq.iloc[num, 0] / 36520, 3))
taglist_freq['freq'] = a
taglist_freq = taglist_freq.rename(columns={0: 'cnt'})

taglist_big = []
taglist_small = []
tagname = taglist_freq._stat_axis.values.tolist()
for num in range(0, len(taglist_freq)):
    if taglist_freq.iloc[num, 1] < 0.2:
        taglist_small.append(tagname[num])
    else:
        taglist_big.append(tagname[num])

tag_main = []
for num in range(0, len(df_taglist)):
    for x in range(0, len(taglist_big)):
        colname = 'tag_' + taglist_big[x]
        if df_taglist.loc[num, colname] == 1:
            tag_main.append(1)
            break
        if x == len(taglist_big) - 1:
            tag_main.append(0)

Counter(tag_main)
tag_drop = []
for x in taglist_small:
    tag_drop.append('tag_' + x)
df_taglist_new = df_taglist.drop(tag_drop, axis=1)

# features
df_features = pd.read_csv('D:/pythonProject/cleaning/var_features.csv', index_col=0, encoding='utf-8')
a = []
for num in range(0, len(features_freq)):
    a.append(round(features_freq.iloc[num, 0] / 36520, 3))
features_freq['freq'] = a
features_freq = features_freq.rename(columns={0: 'cnt'})

a = pd.DataFrame(features)
features = features_freq._stat_axis.values.tolist()
features_big = []
# Steam成就，支持控制器（完全支持控制器，部分支持控制器，手柄），Steam创意工坊，应用内购买
# 远程畅玩（在手机上，在平板上，在电视上），单人，
# 多人（线上玩家对战，在线合作，同屏/分屏玩家对战，同屏/分屏合作，远程同乐，跨平台联机游戏，大型多人在线，局域网玩家对战）
# Steam排行榜
f_remote = []
f_controller = []
f_multiplayer = []

for num in range(0, len(df_features)):
    if df_features.loc[num, 'features_完全支持控制器'] == 1 \
            or df_features.loc[num, 'features_部分支持控制器'] == 1 \
            or df_features.loc[num, 'features_手柄'] == 1:
        f_controller.append(1)
    else:
        f_controller.append(0)

for num in range(0, len(df_features)):
    if df_features.loc[num, 'features_线上玩家对战'] == 1 \
            or df_features.loc[num, 'features_在线合作'] == 1 \
            or df_features.loc[num, 'features_同屏/分屏玩家对战'] == 1 \
            or df_features.loc[num, 'features_同屏/分屏合作'] == 1 \
            or df_features.loc[num, 'features_远程同乐'] == 1 \
            or df_features.loc[num, 'features_跨平台联机游戏'] == 1 \
            or df_features.loc[num, 'features_大型多人在线'] == 1 \
            or df_features.loc[num, 'features_局域网玩家对战'] == 1:
        f_multiplayer.append(1)
    else:
        f_multiplayer.append(0)

for num in range(0, len(df_features)):
    if df_features.loc[num, 'features_在手机上远程畅玩'] == 1 \
            or df_features.loc[num, 'features_在平板上远程畅玩'] == 1 \
            or df_features.loc[num, 'features_在电视上远程畅玩'] == 1:
        f_remote.append(1)
    else:
        f_remote.append(0)

features_big = ['features_Steam成就', 'features_支持控制器', 'features_Steam创意工坊', 'features_应用内购买',
                'features_远程畅玩', 'features_单人', 'features_多人', 'features_Steam排行榜']
features_big = ['features_Steam 成就', 'features_Steam 创意工坊', 'features_应用内购买',
                'features_单人', 'features_Steam 排行榜']

df_features_new = df_features[features_big]
df_features_new['features_远程畅玩'] = f_remote
df_features_new['features_多人'] = f_multiplayer
df_features_new['features_支持控制器'] = f_controller

# genre
df_genre = pd.read_csv('D:/pythonProject/cleaning/var_genre.csv', index_col=0, encoding='utf-8')
a = []
for num in range(0, len(genre_freq)):
    a.append(round(genre_freq.iloc[num, 0] / 36520, 3))
genre_freq['freq'] = a
genre_freq = genre_freq.rename(columns={0: 'cnt'})
genre_freq.to_csv('D:/pythonProject/cleaning/genre_freq.csv', encoding='utf_8_sig')

genre_big = ['genre_独立', 'genre_动作', 'genre_休闲', 'genre_冒险', 'genre_模拟', 'genre_策略']
df_genre_new = df_genre[genre_big]

# no tag, only genre + features
df_raw_data_new = df_raw_data.iloc[:, 0:30]
# 一个月内是否会打折？create y
first_month_discount = []
for num in range(0, len(df_raw_data)):
    if df_raw_data.loc[num, 'first_discount_period'] == 'None':
        first_month_discount.append(0)
        continue
    if int(df_raw_data.loc[num, 'first_discount_period']) <= 30:
        first_month_discount.append(1)
    else:
        first_month_discount.append(0)
Counter(first_month_discount)
df_raw_data_new['first_month_discount'] = first_month_discount

# 加入genre和features
df_raw_data_new = pd.concat([df_raw_data_new, df_genre_new], axis=1)
df_raw_data_new = pd.concat([df_raw_data_new, df_features_new], axis=1)

########################################################################
df = pd.read_csv('D:/pythonProject/cleaning/full_data_raw0331.csv', index_col=0, encoding='utf-8')
df_taglist = df['taglist']
taglist = pd.read_csv('D:/pythonProject/cleaning/var_taglist.csv', index_col=0, encoding='utf-8')
taglist = taglist.drop(a, axis=0)
taglist = taglist.drop(outlier1_index, axis=0)

taglist['tagsum'] = taglist.apply(lambda x: x.sum(), axis=1)
tagsum = list(taglist['tagsum'])








