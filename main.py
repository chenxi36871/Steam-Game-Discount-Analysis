import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import numpy as np

def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def month_mapping(str):
    if str.lower() == 'jan':
        m = 1
    if str.lower() == 'feb':
        m = 2
    if str.lower() == 'mar':
        m = 3
    if str.lower() == 'apr':
        m = 4
    if str.lower() == 'may':
        m = 5
    if str.lower() == 'jun':
        m = 6
    if str.lower() == 'jul':
        m = 7
    if str.lower() == 'aug':
        m = 8
    if str.lower() == 'sep':
        m = 9
    if str.lower() == 'oct':
        m = 10
    if str.lower() == 'nov':
        m = 11
    if str.lower() == 'dec':
        m = 12
    # else:
    #     m = ''
    return m

a = test.split(' ')[1].replace(',','').lower()
if a.lower() == 'oct':
    print(1)
month_mapping(a)

