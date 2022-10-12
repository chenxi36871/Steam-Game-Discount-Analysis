import csv
import datetime as dt
import json
import os
import statistics
import numpy as np
import sklearn as sk
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re
import time

# get app list and appID
url_ori = "https://steamspy.com/api.php"
page = 0 # max page = 52
# url = "https://steamspy.com/api.php?request=all&page=1"
url = url_ori + "?request=all&page=" + str(page)
response = requests.get(url).json()
json_data = pd.DataFrame.from_dict(response,orient='index')

for page in range(0,53):
    page = page +1
    url = url_ori + "?request=all&page=" + str(page)
    response = requests.get(url).json()
    temp = pd.DataFrame.from_dict(response, orient='index')
    json_data = json_data.append(temp, ignore_index=True)  # ignore_index表示不按原来的索引（因为原索引是appid，不是真正的索引）
    # time.sleep(10) # 有时候频繁访问会被阻止

app_list = json_data[['appid', 'name']].sort_values('appid').reset_index(drop=True)

# export disabled to keep consistency across download sessions
app_list.to_csv('D:/pythonProject/app_list.csv', index=False)
json_data.to_csv('D:/pythonProject/steam_spy_all_data.csv', index=False)

# instead read from stored csv
app_list = pd.read_csv('D:/pythonProject/app_list.csv')
steamspy_all = pd.read_csv('D/pythonProject/steam_spy_all_data.csv')

# display first few rows
app_list.head()

# Download Steam Data
# request data from steam website
steamurl_lstg_ori = 'https://store.steampowered.com/app/'
steam_soup = []
for i in range(0,5):
    steamurl_lstg = steamurl_lstg_ori + str(app_list.iloc[i,0])
    r = requests.get(steamurl_lstg, timeout=10)
    temp = BeautifulSoup(r.text, 'lxml')
    steam_soup.append(temp)
    time.sleep(5)

for i in range(0,3):
    steamurl_lstg = steamurl_lstg_ori + str(app_list.iloc[i,0])
    try:
        r = requests.get(steamurl_lstg)
        temp = BeautifulSoup(r.text, 'lxml')
        steam_soup.append(temp)
        time.sleep(10)
    except Exception:
        print('connection error')



    time.sleep(10)

    # print(app_list.iloc[i,0])


soup_now = BeautifulSoup(r.text,'lxml')

#游戏名字
def gamename(soup):
    try:
        a = soup.find(class_="apphub_AppName", id='appHubAppName')
        k = str(a.get_text())
    except:
        k = ''
    return k

#价格（原始价格）original price
def gameprice(soup):
    try:
        test = soup.find(class_='game_purchase_action_bg').find_all(class_='discount_original_price')
        if str(test) == '[]':
            # test1 = soup.find(class_='game_area_purchase_game').find(class_='game_purchase_action').find(
            #     class_='game_purchase_price price')
            test1 = soup.find_all(class_='game_purchase_price price')
            if test1 is None:
                k1 = ''
            # if test1.get_text().replace('\n','').replace('\t','') == '免费开玩':
            #     k1 =
            else:
                k = test1.get_text().replace('\r', '').replace('\n', '').replace('\t', '')
                if k == '免费开玩' or '免费':
                    k1 = k
                else:
                    k1 = re.sub(u"([^\u0030-\u0039\u002e\u00a5\uffe5])", "", k) # \u002e表示小数点
                    # k1 = float(k[1:])
        else:
            test = soup.find(class_='discount_original_price')
            k = test.get_text()
            k1 = re.sub(u"([^\u0030-\u0039\u002e\u00a5\uffe5])", "", k) #\uffe5 表示人民币符号
            # k1 = float(k[2:])
            # print(k1)
    except:
        k1 = ''
    return k1

#游戏描述
def description(soup):
    try:
        a = soup.find(class_="game_description_snippet")
        k = str(a.string).replace('	', '').replace('\n', '').replace('\r', '')
    except:
        k = ''
    return k

#标签列表
def taglist(soup):
    try:
        list1 = []
        a = soup.find_all(class_="app_tag")
        for i in a:
            k = str(i.string).replace('\t', '').replace('\r', '').replace('\n', '')
            if k == '+':
                pass
            else:
                list1.append(k)
        # list1 = str('\n'.join(list1))
    except:
        list1 = ''
    return list1

#近30天总体评价
def reviewsummary(soup):
    try:
        test = a = soup.find_all(class_='nonresponsive_hidden responsive_reviewdesc')
        if len(test) == 1:
            k = 'None'
        else:
            a = soup.find_all(class_="game_review_summary", itemprop='description')
            k = a[0].string
        # a = soup.find(class_="summary column")
        # try:
        #     k = str(a.span.string)
        # except:
        #     k = str(a.text)
    except:
        k = ''
    return k

#所有总体评价
def reviewsummary_forever(soup):
    try:
        test = a = soup.find_all(class_='nonresponsive_hidden responsive_reviewdesc')
        if len(test) == 1:
            a = soup.find(class_='summary column').find_all(class_="game_review_summary", itemprop='description')
            k = a[0].string
        else:
            a = soup.find_all(class_="game_review_summary", itemprop='description')
            k = a[1].string
    except:
        k = ''
    return k

#发行日期
def getdate(soup):
    try:
        a = soup.find(class_="date")
        k = a.string
    except:
        k = ''
    return k

#近30天好评率
#EN
def score(soup):
    try:
        # a = soup.find(class_="user_reviews_summary_row")
        #得到两个结果，近期的和所有的
        a = soup.find_all(class_='nonresponsive_hidden responsive_reviewdesc')
        k = str((a.attrs)['data-tooltip-html'])
        score_str = k.split(' ')[0]
        score = float(score_str[:-1]) / 100
    except:
        score = ''
    return score

#CN
def score_cn(soup):
    try:
        a = soup.find_all(class_='nonresponsive_hidden responsive_reviewdesc')
        if len(a) == 1:
            score = 'None'
        else:
            k = a[0].get_text().replace('\n', '').replace('\t', '').replace('-', '').split(' ')
            score_str = k[6]
            if score_str.isalpha():
                score_str = k[1]
            score = float(score_str[:-1]) / 100
    except:
        score = ''
    return score

# 所有时间的评分
def score_cn_forever(soup):
    try:
        a = soup.find_all(class_='nonresponsive_hidden responsive_reviewdesc')
        if len(a) == 1:
            k = a[0].get_text().replace('\n', '').replace('\t', '').replace('-', '').split(' ')
            score_str = k[3]
            if score_str.isalpha():
                score_str = k[1]
            score = float(score_str[:-1]) / 100
        else:
            k = a[1].get_text().replace('\n', '').replace('\t', '').replace('-', '').split(' ')
            score_str = k[3]
            if score_str.isalpha():
                score_str = k[1]
            score = float(score_str[:-1]) / 100
    except:
        score = ''
    return score

# 过去30天的总用户测评数量
def rating_sample_num(soup):
    try:
        a = soup.find_all(class_='nonresponsive_hidden responsive_reviewdesc')
        if len(a) ==1:
            r = 'None'
        else:
            k = a[0].get_text().replace('\n', '').replace('\t', '').replace('-', '').split(' ')
            r = int(k[4].replace(',', ''))
            # rating_sample_num = int(k.split(' ')[3].replace(',', ''))
    except:
        r = ''
    return r

# 所有时间的总用户测评数量
def rating_sample_num_forever(soup):
    try:
        a = soup.find_all(class_='nonresponsive_hidden responsive_reviewdesc')
        if len(a) == 1:
            k = a[0].get_text().replace('\n', '').replace('\t', '').replace('-', '').split(' ')
        else:
            k = a[1].get_text().replace('\n', '').replace('\t', '').replace('-', '').split(' ')
        if k[-2].isalpha():
            r = int(k[4].replace(',',''))
        else:
            r = int(k[1].replace(',', ''))
        # rating_sample_num = int(k.split(' ')[3].replace(',', ''))
    except:
        r = ''
    return r

#开发商 developer
def developer(soup):
    try:
        test = soup.find_all(id="developers_list")
        k = []
        for a in test:
            k = str(a.get_text().replace('\n', '')).split(',')
        for x in range(1, len(k)):
            k[x] = k[x][1:]
    except:
        k = ''
    return k

#language
def language(soup):
    try:
        test = soup.find(class_="game_language_options")
        k = []
        for idx, tr in enumerate(test.find_all('tr')):
            if idx != 0:
                tds = tr.find_all('td', class_='ellipsis')
                a1 = str(tds[0].contents[0].replace('\r', '').replace('\n', '').replace('\t', ''))
                k.append(a1)
    except:
        k = ''
    return k #list

#支持系统
# def system(soup):
#     try:
#         sys1 = soup.find_all(class_='sysreq_tab')
#         if sys1 == []:
#             sys1 = soup.find(class_='game_area_sys_req_leftCol')  #.find(class_='bb_ul')
#             if sys1 is None:
#                 sys1 = soup.find(class_='game_area_sys_req_full')
#                 k = sys1.get_text()
#             else:
#                 sys1 = soup.find(class_='game_area_sys_req_leftCol').find(class_='bb_ul')
#                 if sys1 is None:
#                     sys1 = soup.find(class_='game_area_sys_req_leftCol')
#                     k = sys1.get_text()
#                 else:
#                     for li in sys1:
#                         k = li.get_text()
#                         break
#             if 'Window' in k or 'windows' in k:
#                 sys = 'win'
#         else:
#             sys = []
#             for i in sys1:
#                 sys.append(i['data-os'])
#                 # print(i['data-os'])
#     except:
#         sys = ''
#     return sys
def system(soup):
    try:
        sys1 = soup.find_all(class_='sysreq_tab')
        if sys1 == []:
            sys = 'win'
        else:
            sys = []
            for i in sys1:
                sys.append(i['data-os'])
                # print(i['data-os'])
    except:
        sys = ''
    return sys

#Genre
def genre(soup):
    try:
        g = soup.find(id='genresAndManufacturer').span.text
    except:
        g = ''
    return g

#features
# single player; online PvP; online Co-op; steam achievements; full controller support; steam trading cards;
# cross-platform multiplayer; in-app purchases; shared/split screen co-op; LAN PvP; steam workshop; Steam Turn Notifications;
# Remote Play on Tablet; Steam Cloud; Remote Play on Phone; Remote Play on Tablet; Remote Play on TV; ...
def features(soup):
    try:
        test = soup.find_all(class_='game_area_details_specs_ctn')
        k = []
        for a in test:
            k.append(a.get_text())
    except:
        k = ''
    return k

app_list = pd.read_csv('D:/pythonProject/app_list.csv')
num = 180
# f2 = open('D:/pythonProject/html_text_result_eg/'+str(app_list.iloc[num,0])+'_'+str(num)+'.txt','r', encoding='utf-8')
f2 = open('D:/pythonProject/html_text_result/'+str(app_list.iloc[num,0])+'_' + str(num) +'.txt','r', encoding='utf-8')
w = str(app_list.iloc[num,0])
test = f2.read()
f2.close()
soup = BeautifulSoup(test, 'lxml')

# 整合数据，建立raw格式化数据
data_raw = app_list
appid = []
name = []
price = []
des = []
tag = []
review_overall = []
review_overall_forever = []
release_date = []
score_30days = []  # 近30天好评率
rating_num_30days = []  # 近30天总测评人数
score_forever = []
rating_num_forever = []
dev = []  # developer
lan = []  # language
sys = []  # supporting system
gen = []  # genre
fea = []  # features
for x in range(45000, 52483):
    f2 = open('D:/pythonProject/html_text_result/' + str(app_list.iloc[x, 0]) +'_' + str(x) + '.txt', 'r', encoding='utf-8')
    w = str(app_list.iloc[x, 0])
    test = f2.read()
    f2.close()
    soup = BeautifulSoup(test, 'lxml')
    appid.append(app_list.iloc[x, 0])
    name.append(gamename(soup))
    price.append(gameprice(soup))
    des.append(description(soup))
    dev.append(developer(soup))
    fea.append(features(soup))
    gen.append(genre(soup))
    lan.append(language(soup))
    rating_num_30days.append(rating_sample_num(soup))
    score_forever.append(score_cn_forever(soup))
    rating_num_forever.append(rating_sample_num_forever(soup))
    release_date.append(getdate(soup))
    review_overall.append(reviewsummary(soup))
    review_overall_forever.append(reviewsummary_forever(soup))
    score_30days.append(score_cn(soup))
    tag.append(taglist(soup))
    sys.append(system(soup))

df_dict = {'gamename': name,
           'gameid': appid,
           'gameprice': price,
           'description': des,
           'taglist': tag,
           'reviewsummary': review_overall,
           'reviewsummary_forever': review_overall_forever,
           'release_date': release_date,
           'score_30days': score_30days,
           'rating_sample_num_30days': rating_num_30days,
           'score_forever': score_forever,
           'rating_sample_num_forever': rating_num_forever,
           'developer': dev,
           'language': lan,
           'system': sys,
           'genre': gen,
           'features': fea}
df = pd.DataFrame(df_dict)
df.to_csv('D:/pythonProject/raw_data_10.csv',encoding='utf_8_sig')


gamename(soup)
gameprice(soup)
description(soup)  # 中文
taglist(soup)
reviewsummary(soup)
getdate(soup)  # '2000 年 11 月 1 日' ## 怎么转换成日期格式
score_cn(soup)  # 过去30天的好评率
rating_sample_num(soup)  # 过去30天的总测评数量
developer(soup)
language(soup)
system(soup)
genre(soup)
features(soup)



#获取评论?
# def getreviews(ID):
#     r1 = requests.get(
#         'https://store.steampowered.com/appreviews/%s?cursor=*&day_range=30&start_date=-1&end_date=-1&date_range_type=all&filter=summary&language=schinese&l=schinese&review_type=all&purchase_type=all&playtime_filter_min=0&playtime_filter_max=0&filter_offtopic_activity=1'%str(ID),headers=headers,timeout=10)
#     soup = BeautifulSoup(r1.json()['html'], 'lxml')
#     a = soup.findAll(class_="content")
#     list1 = []
#     for i in a:
#         list1.append(i.text.replace('	', '').replace('\n', '').replace('\r', '').replace(' ', ','))
#     k=str('\n'.join(list1))
#     return k
# getreviews(730)

# 爬取历史价格
# 根据gamename来获取url
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import numpy as np

app_list = pd.read_csv('D:/pythonProject/app_list.csv')

# f2 = open('D:/pythonProject/html_text_result/'+str(app_list.iloc[i,0])+'_'+str(i)+'.txt','r', encoding='utf-8')
# f2 = open('D:/pythonProject/html_text_result/'+str(app_list.iloc[num,0])+'.txt','r', encoding='utf-8')
# w = str(app_list.iloc[num,0])
# test = str(f2.readlines())
# f2.close()
# temp_soup = BeautifulSoup(test, 'lxml')

# 合并52483条数据并进行初步清洗
df_read = pd.DataFrame()
for n in range(1,11):
    path = 'D:/pythonProject/raw_data_' + str(n) + '.csv'
    df_read_temp = pd.read_csv(path).iloc[:,1:]
    df_read = pd.concat([df_read, df_read_temp], ignore_index=True)
df_read.to_csv('D:/pythonProject/raw_data_full.csv',encoding='utf_8_sig')


def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

df_read = pd.read_csv('D:/pythonProject/raw_data_full.csv',encoding='utf_8_sig')
format_release_date = df_read.iloc[:, 8]
format_date_temp = []
for x in range(0, 52483):
    try:
        test = format_release_date[x]
        if is_contain_chinese(test):
            year = int(test.split(' ')[0])
            month = int(test.split(' ')[2])
            date = int(test.split(' ')[4])
            format_date_temp.append(datetime(year, month, date))
        else:
            year = int(test.split(' ')[2])
            month = int(month_mapping(test.split(' ')[1].replace(',','').lower()))
            date = int(test.split(' ')[0])
            format_date_temp.append(date(year, month, date))
    except:
        format_date_temp.append('')

df_read['format_release_date'] = format_date_temp
df_read_temp = df_read
# df_read_clear = df_read_temp.drop(df_read_temp[df_read_temp['format_release_date'] < datetime(2016, 1, 1)].index)
mark = []
for n in range(0, 52483):
    if df_read_temp.iloc[n, 18] < datetime(2016, 1, 1) or pd.isnull(np.datetime64(str(df_read_temp.iloc[n, 18]))):
        mark.append(1)
    else:
        mark.append(0)
df_read_temp['mark'] = mark
df_read_temp = df_read_temp.drop(df_read_temp[df_read_temp['mark'] == 1].index)
# 保留release date在2016-01-01之后并且游戏名不为空的 ---raw_data_full_clear1.csv
df_read_temp.to_csv('D:/pythonProject/raw_data_clear1.csv', encoding='utf_8_sig')

# 从isthereanydeal网站获取游戏的历史价格和发行价格
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import time

df_read_clear = pd.read_csv('D:/pythonProject/raw_data_clear1.csv', index_col=0).drop(['mark'], axis=1)
df_read_clear = df_read_clear.drop(['release_date'], axis=1)

headers = {
    'connection': 'close',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    # 'Accept-Language': 'zh-EN',
    'Cache-Control': 'max-age=0',
    # 'Connection': 'keep-alive',
    # 'Cookie':'browserid=2551845692279885660; timezoneOffset=28800,0; _ga=GA1.2.789263067.1635341841; _gid=GA1.2.1599620076.1647264225; steamMachineAuth76561198866429282=7E08A439ED8E8B5BB77BF25B1971C5DFDFE7C8EB; Steam_Language=english; steamRememberLogin=76561198866429282%7C%7Cd05744be103142d0a7aced2d106e0be8; lastagecheckage=1-0-1999; sessionid=6168bb2d750961d6090e4438; app_impressions=1367550@1_241_4_strategy_201|1203220@1_7_7_230_150_1|730@1_7_7_230_150_1|1426210@1_7_7_230_150_1|1245620@1_7_7_230_150_1|359550@1_7_7_230_150_1|1446780@1_7_7_230_150_1|1174180@1_7_7_230_150_1|578080@1_7_7_230_150_1; steamLoginSecure=76561198866429282%7C%7C8C4788C65FE9C3A101AD8C8CA0EFDF83AF4EE9B1; recentapps=%7B%221426210%22%3A1647402535%2C%2210%22%3A1647368366%2C%22289070%22%3A1647366549%2C%2248000%22%3A1647358907%2C%221245620%22%3A1647357040%2C%221329410%22%3A1647355898%2C%221335200%22%3A1647320954%2C%221056690%22%3A1647318535%2C%221085660%22%3A1647318265%2C%22632360%22%3A1647317685%7D',
    # 'Host': 'store.steampowered.com',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
}

release_price_real = []
df_history_price = pd.DataFrame(index=np.arange(5000))
df_history_date = pd.DataFrame(index=np.arange(5000))
df_full_price = pd.DataFrame(index=np.arange(5000))

num = 25000
error_list = []

# 25000~30000

while num < 30000:
    try:
        url_proxy = 'http://proxy.httpdaili.com/apinew.asp?ddbh=2540800152610572559'
        r_proxy = requests.get(url_proxy).text
        proxy_temp = r_proxy.replace('\n', '').replace('\r', ',').split(',')
        proxy = {'https': proxy_temp[0]}
        print(proxy)
        #### check proxy
        # response = requests.get("https://httpbin.org/ip", proxies=proxy)
        # print(response.text)
        # response = requests.get("https://httpbin.org/ip")
        # print(response.text)
        game_name = str(re.sub(r"[^a-zA-Z0-9]", "", df_read_clear.iloc[num, 1].lower()))
        if bool(re.search(r'\d', game_name)):
            game_name = game_name.replace('1', 'i')
            game_name = game_name.replace('2', 'ii')
            game_name = game_name.replace('3', 'iii')
            game_name = game_name.replace('4', 'iv')
            game_name = game_name.replace('5', 'v')
            game_name = game_name.replace('6', 'vi')
            game_name = game_name.replace('7', 'vii')
            game_name = game_name.replace('8', 'viii')
            game_name = game_name.replace('9', 'ix')
            game_name = game_name.replace('9', 'x')
        url_history = 'https://isthereanydeal.com/game/' + game_name + '/history/?country=CN&currency=CNY&shop%5B%5D=steam&generate=Select+Stores'
        r_history = requests.get(url_history, headers=headers, proxies=proxy, verify=False, timeout=(5, 15))
        print(r_history.status_code)
        print(num)
        soup_history = BeautifulSoup(r_history.text, 'lxml')
        # history price ￥ (in yuan)
        test = soup_history.find_all(class_='lg2__price')

        if test == []:
            game_name = str(re.sub(r"[^a-zA-Z0-9]", "", df_read_clear.iloc[num, 1].lower().replace('the', '')))
            if bool(re.search(r'\d', game_name)):
                game_name = game_name.replace('1', 'i')
                game_name = game_name.replace('2', 'ii')
                game_name = game_name.replace('3', 'iii')
                game_name = game_name.replace('4', 'iv')
                game_name = game_name.replace('5', 'v')
                game_name = game_name.replace('6', 'vi')
                game_name = game_name.replace('7', 'vii')
                game_name = game_name.replace('8', 'viii')
                game_name = game_name.replace('9', 'ix')
                game_name = game_name.replace('9', 'x')
            url_history = 'https://isthereanydeal.com/game/' + game_name + '/history/?country=CN&currency=CNY&shop%5B%5D=steam&generate=Select+Stores'
            r_history = requests.get(url_history, headers=headers, proxies=proxy, verify=False, timeout=(5, 15))
            print(r_history.status_code)
            print(num)

            soup_history = BeautifulSoup(r_history.text, 'lxml')
            # history price ￥ (in yuan)
            test = soup_history.find_all(class_='lg2__price')
        price = []  # 全部价格：原价+打折价格
        for t in test:
            price.append(float(t.get_text()[1:]))
            # print(i.get_text()[1:])
        seq = ['' for _ in range(5000 - len(price))]
        price.extend(seq)

        release_price_real.append(price[-2])

        price_actual = []  # 实际价格（打折价格）
        for t in range(1, len(price), 2):
            price_actual.append(price[t])
        seq = ['' for _ in range(5000 - len(price_actual))]
        price_actual.extend(seq)
        # print(i)

        # history time
        test1 = soup_history.find_all(class_='lg2__time-rel')
        history_date = []
        for t in test1:
            # temp = datetime.strptime(i.get_text()[:10], "%Y-%m-%d").date()
            temp = str(t.get_text()[:10])
            history_date.append(temp)
        seq = ['' for _ in range(5000 - len(history_date))]
        history_date.extend(seq)

        df_history_price[str(df_read_clear.iloc[num, 0])] = price_actual
        df_history_date[str(df_read_clear.iloc[num, 0])] = history_date
        df_full_price[str(df_read_clear.iloc[num, 0])] = price
        # file = open('D:/pythonProject/history_price/' + str(df_read_clear.iloc[num, 0]) + '_' + game_name + '.txt', 'w',
        #             encoding='utf-8')
        # file.write(r_history.text)
        # file.close()
        num = num + 1
        time.sleep(1)
    except:
        error_list.append(df_read_clear.iloc[num, 0])
        num = num+1
        time.sleep(1)

df_history_price.to_csv('D:/pythonProject/history_price/history_price_25000-30000.csv', encoding='utf-8')
df_history_date.to_csv('D:/pythonProject/history_price/history_date_25000-30000.csv', encoding='utf-8')
df_full_price.to_csv('D:/pythonProject/history_price/full_price_25000-30000.csv', encoding='utf-8')
df_error_list = pd.DataFrame(error_list)
df_error_list.to_csv('D:/pythonProject/history_price/error_list_25000-30000.csv', encoding='utf-8')
df_release_price_real = pd.DataFrame(release_price_real)
df_release_price_real.to_csv('D:/pythonProject/history_price/release_price_real_25000-30000.csv', encoding='utf-8')

# 从steamspy爬取用户人数、游玩时长等相关信息
# owners: owners of this application on Steam as a range
# average_forever: average playtime since March 2009. In minutes
# average_2weeks: average playtime in the last two weeks. In minutes
# median_forever: median playtime since March 2009. In minutes
# median_2weeks: median playtime in the last two weeks. In minutes
# ccu: peak CCU (concurrent user 同时在线人数) yesterday
import requests
import pandas as pd
import time

df_read_clear = pd.read_csv('D:/pythonProject/raw_data_clear1.csv', index_col=0).drop(['mark'], axis=1)
df_steamspy = pd.DataFrame(columns=['index', 'appid', 'owner', 'avg_forever', 'avg_2weeks', 'median_forever', 'median_2weeks', 'ccu'])

num = 28500
while num < 29300:
    try:
        try:
            appid = df_read_clear.iloc[num, 2]
            url = 'https://steamspy.com/api.php?request=appdetails&appid=' + str(appid)
            # parameters = {"request": "appdetails", "appid": appid}
            url_proxy = 'http://proxy.httpdaili.com/apinew.asp?ddbh=2542653972570572559'
            r_proxy = requests.get(url_proxy).text
            proxy_temp = r_proxy.replace('\n', '').replace('\r', ',').split(',')
            proxy = {'https': proxy_temp[0]}
            print(proxy)
            print(num)
            r_spy = requests.get(url, verify=False, proxies=proxy, timeout=(5, 15))
            print(r_spy.status_code)
            result = r_spy.json()
            df_steamspy.loc[num] = [df_read_clear.iloc[num, 0], appid, result['owners'], result['average_forever'],
                                    result['average_2weeks'], result['median_forever'], result['median_2weeks'],
                                    result['ccu']]
            num = num+1
            time.sleep(1)
        except:
            appid = df_read_clear.iloc[num, 2]
            url = 'https://steamspy.com/api.php?request=appdetails&appid=' + str(appid)
            # parameters = {"request": "appdetails", "appid": appid}
            url_proxy = 'http://proxy.httpdaili.com/apinew.asp?ddbh=2542653972570572559'
            r_proxy = requests.get(url_proxy).text
            proxy_temp = r_proxy.replace('\n', '').replace('\r', ',').split(',')
            proxy = {'https': proxy_temp[1]}
            print(proxy)
            print(num)
            r_spy = requests.get(url, verify=False, proxies=proxy, timeout=(5, 15))
            print(r_spy.status_code)
            result = r_spy.json()
            df_steamspy.loc[num] = [df_read_clear.iloc[num, 0], appid, result['owners'], result['average_forever'],
                                    result['average_2weeks'], result['median_forever'], result['median_2weeks'],
                                    result['ccu']]
            num = num+1
            time.sleep(1)
    except:
        df_steamspy.loc[num] = [df_read_clear.iloc[num, 0], appid, '', '', '', '', '', '']
        time.sleep(1)
        print(num)
        num = num+1
    if num % 100 == 0 or num ==44169:
        download_path = 'D:/pythonProject/steam_spy/'
        file_name = 'raw_data_steamspy_' + str(num) + '.csv'
        df_steamspy.to_csv(download_path + file_name, encoding='utf-8')
        df_steamspy = pd.DataFrame(columns=['index', 'appid', 'owner', 'avg_forever', 'avg_2weeks',
                                            'median_forever', 'median_2weeks', 'ccu'])

df_steamspy.to_csv('D:/pythonProject/raw_data_steamspy_0~5000.csv', encoding='utf-8')



# positive: sample_num * score