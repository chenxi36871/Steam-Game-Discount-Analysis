import requests
import time
import pandas as pd
import socket

headers = {
    'connection': 'close',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    # 'Accept-Language': 'zh-EN',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    # 'Cookie':'browserid=2551845692279885660; timezoneOffset=28800,0; _ga=GA1.2.789263067.1635341841; _gid=GA1.2.1599620076.1647264225; steamMachineAuth76561198866429282=7E08A439ED8E8B5BB77BF25B1971C5DFDFE7C8EB; Steam_Language=english; steamRememberLogin=76561198866429282%7C%7Cd05744be103142d0a7aced2d106e0be8; lastagecheckage=1-0-1999; sessionid=6168bb2d750961d6090e4438; app_impressions=1367550@1_241_4_strategy_201|1203220@1_7_7_230_150_1|730@1_7_7_230_150_1|1426210@1_7_7_230_150_1|1245620@1_7_7_230_150_1|359550@1_7_7_230_150_1|1446780@1_7_7_230_150_1|1174180@1_7_7_230_150_1|578080@1_7_7_230_150_1; steamLoginSecure=76561198866429282%7C%7C8C4788C65FE9C3A101AD8C8CA0EFDF83AF4EE9B1; recentapps=%7B%221426210%22%3A1647402535%2C%2210%22%3A1647368366%2C%22289070%22%3A1647366549%2C%2248000%22%3A1647358907%2C%221245620%22%3A1647357040%2C%221329410%22%3A1647355898%2C%221335200%22%3A1647320954%2C%221056690%22%3A1647318535%2C%221085660%22%3A1647318265%2C%22632360%22%3A1647317685%7D',
    'Host': 'store.steampowered.com',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36'
}


app_list = pd.read_csv('D:/pythonProject/app_list.csv')
steamurl_lstg_ori = 'https://store.steampowered.com/app/'
# steamurl_lstg = 'https://store.steampowered.com/app/460930/Tom_Clancys_Ghost_Recon_Wildlands/'
i = 180

while i == 6332:
    try:
        steamurl_lstg = steamurl_lstg_ori + str(app_list.iloc[i, 0])
        requests.adapters.DEFAULT_RETRIES = 5
        r = requests.get(steamurl_lstg, headers=headers, verify=False, timeout=(5,15))
        print(r.status_code)
        file = open('D:/pythonProject/html_text_result/' + str(app_list.iloc[i, 0]) + '_' + str(i) + '.txt', 'w',
                    encoding='utf-8')
        file.write(r.text)
        file.close()
        i = i + 1
        # time.sleep(1)
    except BaseException or ConnectionError or socket.timeout:
        ###two proxy
        steamurl_lstg = steamurl_lstg_ori + str(app_list.iloc[i, 0])
        requests.adapters.DEFAULT_RETRIES = 5
        s = requests.session()
        s.keep_alive = False
        s.trust_env = False
        s.proxies = {'https': 'https://119.123.175.200:9797'}
        # s.proxies = {'https': 'https://42.180.225.146:10057'}
        # s.proxies = {'https': 'https://140.250.146.82:13456'}
        s.headers = headers
        r = s.get(steamurl_lstg, verify=False, timeout=(5,15))
        print(r.status_code)  # success=200
        file = open('D:/pythonProject/html_text_result/' + str(app_list.iloc[i, 0]) + '_' + str(i) + '.txt', 'w',
                    encoding='utf-8')
        file.write(r.text)
        file.close()
        i = i + 1
        # time.sleep(2)
    # time.sleep(5)





