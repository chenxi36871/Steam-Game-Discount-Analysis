# Steam Game Discount Analysis

## Purpose
- For a newly released game on Steam platform, predict whether there'll be discount within the first month after it's released.
- Predict how much time it would take for a game to reach its 50% discount

## Data Source
There are in total 52843 observations/records of games.
- Basic information of the game (name, genre, features, release date, rating, etc.) comes from [***Steam official website***](https://store.steampowered.com/). I scraped them using Steam API.
- Price histroy of the games comes from [***IsThereAnyDeal website***](https://isthereanydeal.com/).
- Data about player information (play time, ccu, etc.) comes from [***SteamSpy website***](https://steamspy.com/)

## Outline 
Link directly goes to related Python code.

[web scraper to get raw data](https://github.com/chenxi36871/Steam-game-discount-analysis/blob/70613d5cce6c9b32ed697b98f914c79304f81fc2/pachong_try.py)

[data cleaning and preprocessing](https://github.com/chenxi36871/Steam-game-discount-analysis/blob/main/cleaning.py)

[visulization](https://github.com/chenxi36871/Steam-game-discount-analysis/blob/main/data%20visualization.py)

[modeling: predict first discount](https://github.com/chenxi36871/Steam-game-discount-analysis/blob/main/model_first%20discount.py)

[modeling: predict how much time it takes to reach 50% discount](https://github.com/chenxi36871/Steam-game-discount-analysis/blob/main/model3-50.py)


[final report (in Chinese)](https://github.com/chenxi36871/Steam-game-discount-analysis/blob/main/%E5%9F%BA%E4%BA%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9A%84%E7%94%B5%E5%AD%90%E6%B8%B8%E6%88%8F%E5%B8%82%E5%9C%BA%E7%9A%84%E4%BF%83%E9%94%80%E7%AD%96%E7%95%A5%E9%A2%84%E6%B5%8B%E5%92%8C%E5%BD%B1%E5%93%8D%E5%9B%A0%E7%B4%A0%E7%A0%94%E7%A9%B6_%E6%9D%8E%E6%99%A8%E8%8C%9C2018110760.pdf)
