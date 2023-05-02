import telebot
import pandas as pd
from PIL import Image
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from implicit.als import AlternatingLeastSquares as ALS
import sqlite3

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For implicit ALS


load_dotenv()
TOKEN = (os.getenv('TOKEN'))
bot = telebot.TeleBot(TOKEN)
model = ALS().load(os.getenv('als_model'))

nickname = None


b_games = pd.read_feather(os.getenv('bgg_boardgames_top_2000'))
ratings = pd.read_feather(os.getenv('bgg_ratings_top_2000'))

users_inv_mapping = dict(enumerate(ratings['nickname'].unique()))
users_mapping = {v: k for k, v in users_inv_mapping.items()}

item_titles = pd.Series(ratings['title'].values, index=ratings['boardgame_id']).to_dict()

title_items = ratings.groupby('title')['boardgame_id'].agg(list)


items_inv_mapping = dict(enumerate(ratings['boardgame_id'].unique()))
items_mapping = {v: k for k, v in items_inv_mapping.items()}

def get_similar_games(title, model, message, N=5, idx=0, 
                      title_items=title_items, item_titles=item_titles, items_mapping=items_mapping, items_inv_mapping=items_inv_mapping):
    item_ids = title_items.at[title]
    if len(item_ids) == 1:
        item_id = item_ids[0]
    else:
        item_id = item_ids[idx]
    
    col_id = items_mapping[item_id]
    similar = model.similar_items(col_id, N=N+1)
    similar = pd.DataFrame({'col_id':similar[0], 'similarity':similar[1]})
    similar['item_id'] = similar['col_id'].map(items_inv_mapping.get)
    similar['title'] = similar['item_id'].map(item_titles.get)

    for i in range(1,len(similar["item_id"])):
        url=b_games.loc[b_games["boardgame_id"]==similar['item_id'][i],"image_link"][:1].values[0]
        img=Image.open(requests.get(url,stream=True).raw)
        bot.send_photo(message.chat.id, 
            img,
            caption = (
            f'''
        Номер: {similar.index[i]}. 
        <b>Название</b>: <u style="color:red">{similar.iloc[i].title}</u>
        <b>Рейтинг</b>: {b_games.loc[b_games["boardgame_id"]==similar['item_id'][i],"average_rating"].values[0].astype(float).round(2)}
       '''# <b>Минимальное количество игроков</b>: {popularGames.iloc[i].minplayers}
       # <b>Максимальное количество игроков</b>: {popularGames.iloc[i].maxplayers}
       #<b>Время игры</b>: {popularGames.iloc[i].maxplaytime} минут
        #<b>Минимальный возраст для игры</b>: {popularGames.iloc[i].age}
        #<b>Механики</b>: {popularGames.iloc[i].mechanics.replace('|', ', ')}
        #<b>Категория</b>: {popularGames.iloc[i].category.replace('|', ', ')}'''
        ),
        parse_mode='html')


def get_categories():
    categories_set=set()
    for lists in b_games['category'].str.split('|').dropna():
        for category in lists:
            categories_set.add(category)
    return categories_set

def popular_games(df, message, n=10):  
    popularGames = df
    
    def weighted_rate(x):
        v=x["users_rated"]
        R=x["average_rating"]
        
        return ((v*R) + (m*C)) / (v+m)
    
    C=popularGames["average_rating"].mean()
    m=popularGames["users_rated"].quantile(0.90)
     
    popularGames["Popularity"]=popularGames.apply(weighted_rate,axis=1)
    popularGames=popularGames.sort_values(by="Popularity",
                                          ascending=False, 
                                          ignore_index=True).head(n+100).sample(n)

    for i in range(len(popularGames["boardgame_id"])):
        url=popularGames.iloc[i]['image_link'] #5 столбец - это image_link
        img=Image.open(requests.get(url,stream=True).raw)
        bot.send_photo(message.chat.id, 
            img,
            caption = (
            f'''
        Номер в рейтинге: {popularGames.index[i]+1}. 
        <b>Название</b>: <u style="color:red">{popularGames.iloc[i].title}</u>
        <b>Рейтинг</b>: {popularGames.iloc[i].average_rating.astype(float).round(2)}
        <b>Минимальное количество игроков</b>: {popularGames.iloc[i].minplayers}
        <b>Максимальное количество игроков</b>: {popularGames.iloc[i].maxplayers}
        <b>Время игры</b>: {popularGames.iloc[i].maxplaytime} минут
        <b>Минимальный возраст для игры</b>: {popularGames.iloc[i].age}
        <b>Механики</b>: {popularGames.iloc[i].mechanics.replace('|', ', ')}
        <b>Категория</b>: {popularGames.iloc[i].category.replace('|', ', ')}
        '''),
        parse_mode='html')


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Привет!', parse_mode='html')

@bot.message_handler(commands=['top'])
def top(message):
    msg = bot.send_message(message.chat.id, 
                           'Сколько игр тебе показать? 🙂 (пришли цифру)', 
                           parse_mode='html')
    bot.register_next_step_handler(msg, number)
def number(message):
    try:
            popular_games(b_games, message, n=int(message.text))
    except ValueError:
            bot.send_message(message.chat.id, 'Не понимаю 🙁', parse_mode='html')

@bot.message_handler(commands=['random'])
def random(message):
    bot.send_message(message.chat.id, 'Попробуй поиграть в эту 😃', parse_mode='html')
    sample = b_games.sample()
    sample.average_rating = sample.average_rating.astype(float).round(2)
    img=Image.open(requests.get(sample['image_link'].values[0],stream=True).raw)
    bot.send_photo(message.chat.id, 
        img, 
        caption = 
        f'''
        <b>Название</b>: <u>{sample.title.values[0]}</u>
        <b>Рейтинг</b>: {sample.average_rating.values[0]}
        <b>Минимальное количество игроков</b>: {sample.minplayers.values[0]}
        <b>Максимальное количество игроков</b>: {sample.maxplayers.values[0]}
        <b>Время игры</b>: {sample.maxplaytime.values[0]} минут
        <b>Минимальный возраст для игры</b>: {sample.age.values[0]}
        <b>Механики</b>: {sample.mechanics.values[0].replace('|', ', ')}
        <b>Категория</b>: {sample.category.values[0].replace('|', ', ')}
        ''',
        parse_mode='html'
        )
    
@bot.message_handler(commands=['description'])
def description(message):
    msg = bot.send_message(message.chat.id, 
                           'Описание какой настольной игру ты хочешь узнать? 🙂 (работает только на английском языке)', 
                           parse_mode='html')
    bot.register_next_step_handler(msg, game_title)
def game_title(message):
    try:
        output = BeautifulSoup(b_games.loc[b_games['title'].str.contains(message.text, case = False)]['description'].values[0], features="lxml").get_text()
        bot.send_message(message.chat.id, 
            f''' 
            <b>{b_games.loc[b_games['title'].str.contains(message.text, case = False)]['title'].values[0]}</b>
            {output}''',
            parse_mode='html')
    except:
        bot.send_message(message.chat.id, 'Извини, либо нет такой игры либо что-то пошло не так🙂', parse_mode='html')


@bot.message_handler(commands=['item_recs'])
def item_recs(message):
    msg = bot.send_message(message.chat.id, 'Помогу найти похожую игру. Введи название 🙂 (работает только на английском языке)', parse_mode='html')
    bot.register_next_step_handler(msg, item)
def item(message):
    try:
        title = b_games.loc[b_games['title'].str.contains(message.text, case = False)]['title'].values[0]
        bot.send_message(message.chat.id, 
            get_similar_games(title, model, message),
            parse_mode='html')
    except:
        bot.send_message(message.chat.id, 'Извини, либо нет такой игры либо что-то пошло не так🙂', parse_mode='html')


@bot.message_handler(commands=['user_recs'])
def user_recs(message):
   
    conn = sqlite3.connect('database_board_games.sql')
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS users (id int auto_increment primary key, nickname varchar(50), pass varchar(50))')
    conn.commit()
    cur.close()
    conn.close()
    
    msg = bot.send_message(message.chat.id, 'Чтобы дать персональные рекомендации, давайте поймем кто вы 🙂 \nВведите никнейм', parse_mode='html')
    bot.register_next_step_handler(msg, define_user)

def define_user(message):
    pass
    
def user_name(message):
    global nickname 
    nickname = message.text.strip()
    bot.send_message(message.chat.id, 'Введите пароль 🙂', parse_mode='html')
    bot.register_next_step_handler(message, user_pass)
    
def user_pass(message):
    password = message.text.strip()
    bot.send_message(message.chat.id, 'Введите пароль 🙂', parse_mode='html')
    bot.register_next_step_handler(message, user_pass)

    conn = sqlite3.connect('database_board_games.sql')
    cur = conn.cursor()
    cur.execute("INSERT INTO users(nickname, pass) VALUES ('%s', '%s')" % (nickname, password))
    conn.commit()
    cur.close()
    conn.close()

    markup = telebot.types.InlineKeyboardMarkup()
    markup.add(telebot.types.InlineKeyboardButton('Список пользователей', callback_data='users'))
    bot.send_message(message.chat.id, 'Пользователь зарегистрирован 🙂', parse_mode='html', reply_markup=markup)


@bot.callback_query_handler(func = lambda call: True)
def callback(call):
  conn = sqlite3.connect('database_board_games.sql')
  cur = conn.cursor()
  
  cur.execute("SELECT * FROM users")
  users = cur.fetchall()
  
  info = ''
  for el in users:
      info += f'Имя: {el[1]}\n'
  
  cur.close()
  conn.close()
  
  bot.send_message(call.message.chat.id, info)


bot.polling(non_stop=True)

















