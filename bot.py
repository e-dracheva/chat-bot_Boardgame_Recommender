import telebot
import pandas as pd
from PIL import Image
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from implicit.als import AlternatingLeastSquares as ALS
import sqlite3
from telebot.types import WebAppInfo
import numpy as np
import scipy.sparse as sp

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For implicit ALS


load_dotenv()
TOKEN = (os.getenv('TOKEN'))
bot = telebot.TeleBot(TOKEN)
model = ALS().load('models/als_model.npz')

nickname = None


b_games = pd.read_feather('datasets/bgg_boardgames_top_2000.feather')
ratings = pd.read_feather('datasets/bgg_ratings_top_2000.feather')

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

    for i in range(1, N+1):
        url=b_games.loc[b_games["boardgame_id"]==similar['item_id'][i],"image_link"].values[0]
        img=Image.open(requests.get(url,stream=True).raw)
        markup = telebot.types.InlineKeyboardMarkup()
        btn1 = telebot.types.InlineKeyboardButton('Посмотреть описание', callback_data='description')
        btn2 = telebot.types.InlineKeyboardButton('Купить', web_app=WebAppInfo(url = 'https://e-dracheva.github.io/'))
        markup.add(btn1, btn2)
        bot.send_photo(message.chat.id, 
            img,
            caption = (
            f'''<b>{similar.iloc[i].title}</b>'''),
        parse_mode='html',
        reply_markup = 	markup)


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
        url=popularGames.iloc[i]['image_link']
        img=Image.open(requests.get(url,stream=True).raw)
        markup = telebot.types.InlineKeyboardMarkup()
        btn1 = telebot.types.InlineKeyboardButton('Посмотреть описание', callback_data='description')
        btn2 = telebot.types.InlineKeyboardButton('Купить', web_app=WebAppInfo(url = 'https://e-dracheva.github.io/'))
        markup.add(btn1, btn2)
        bot.send_photo(message.chat.id, 
            img,
            caption = (
            f'''
        <b>{popularGames.iloc[i].title}</b>
        '''),
        parse_mode='html',
        reply_markup = 	markup)


def get_coo_matrix(df, 
                   user_col='nickname', 
                   item_col='boardgame_id', 
                   weight_col=None, 
                   users_mapping=users_mapping, 
                   items_mapping=items_mapping):
    if weight_col is None:
        weights = np.ones(len(df), dtype=np.float32)
    else:
        weights = df[weight_col].astype(np.float32)

    interaction_matrix = sp.coo_matrix((
        weights, 
        (
            df[user_col].map(users_mapping.get), 
            df[item_col].map(items_mapping.get)
        )
    ))
    return interaction_matrix            


def generate_personal_recs(message, user, model=model, matrix=get_coo_matrix(ratings).tocsr(), N=5, 
                           users_mapping=users_mapping, items_inv_mapping=items_inv_mapping):

    user_id = users_mapping[user]
    recs = model.recommend(user_id, 
                               matrix[user_id], 
                               N=N, 
                               filter_already_liked_items=True)

    recs = pd.DataFrame(recs).T.rename(columns = {0: 'col_id', 1: 'similarity'})
    recs['item_id'] = recs['col_id'].map(items_inv_mapping.get)
    recs['title'] = recs['item_id'].map(item_titles.get)
    for i in range(N):
        url=b_games.loc[b_games["boardgame_id"]==recs['item_id'][i],"image_link"].values[0]
        img=Image.open(requests.get(url,stream=True).raw)
        markup = telebot.types.InlineKeyboardMarkup()
        btn1 = telebot.types.InlineKeyboardButton('Посмотреть описание', callback_data='description')
        btn2 = telebot.types.InlineKeyboardButton('Купить', web_app=WebAppInfo(url = 'https://e-dracheva.github.io/'))
        markup.add(btn1, btn2)
        bot.send_photo(message.chat.id, 
            img,
            caption = (
            f'''<b>{recs.iloc[i].title}</b>'''),
        parse_mode='html',
        reply_markup = 	markup)


@bot.message_handler(commands=['users'])
def users(message):
    users = ratings['nickname'].sample(n=5).values.tolist()
    bot.send_message(message.chat.id, str(users))


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 
                     f'''
                     Привет! \nЭтот бот умеет рекомендовать тебе различные настольные игры 🎲 \n
<b>Выбери, как ты хочешь получить рекомендации: </b>
    
    🔝 {"/top"} посмотреть самое популярное
    
    👯‍♀ {"/item_recs"} найти похожую игру. 
    
    🦹‍♀ Если у тебя есть учетка, то ты можешь посмотреть персональные рекомендации - {"/user_recs"}
                         
    🎱 Также можно просто запросить случайную игру - {"/random"}. 
    
    🔎 А если тебя интересует определенная настольная игра, то можно ее найти так - {"/search"}
                         
    Учти, что бот работает пока только на английском языке 🇬🇧
    
    Также любую игру можно купить нажав на кнопку "Купить" и заполнить платежные данные 💵
                     ''', 
                     parse_mode='html',
                     )


@bot.message_handler(commands=['top'])
def top(message):
    msg = bot.send_message(message.chat.id, 
                           'Сколько игр тебе показать? 🙂 (пришли цифру)', 
                           parse_mode='html')
    bot.register_next_step_handler(msg, callback=number)
    
def number(message):
        try:
                popular_games(b_games, message, n=int(message.text))
        except ValueError:
                bot.reply_to(message, 'Не понимаю 🙁', parse_mode='html')


@bot.message_handler(commands=['random'])
def random(message):
    bot.send_message(message.chat.id, 'Попробуй поиграть в эту 😃', parse_mode='html')
    sample = b_games.sample()
    sample.average_rating = sample.average_rating.astype(float).round(2)
    img=Image.open(requests.get(sample['image_link'].values[0],stream=True).raw)
    markup = telebot.types.InlineKeyboardMarkup()
    btn1 = telebot.types.InlineKeyboardButton('Посмотреть описание', callback_data='description')
    btn2 = telebot.types.InlineKeyboardButton('Купить', web_app=WebAppInfo(url = 'https://e-dracheva.github.io/'))
    markup.add(btn1, btn2)
    bot.send_photo(message.chat.id, 
        img, 
        caption = 
        f'''
        {sample.title.values[0]}
        ''',
        parse_mode='html',
        reply_markup = 	markup)

@bot.message_handler(commands=['search'])
def search(message):
    msg = bot.send_message(message.chat.id, 'Введи игру, которую хочешь найти 🙂(работает только на английском языке)', parse_mode='html')
    bot.register_next_step_handler(msg, searching)
def searching(message):
    try:
        search_result = b_games.loc[b_games['title'].str.contains(message.text, case = False)]
        search_result_dict = dict(enumerate(search_result['title'].values))
        bot.send_message(message.chat.id, 'Вот что нашлось 🙂')
        for game in search_result_dict:
              img=Image.open(requests.get(search_result[search_result['title']==search_result_dict[game]]['image_link'].values[0], stream=True).raw)
              markup = telebot.types.InlineKeyboardMarkup()
              btn1 = telebot.types.InlineKeyboardButton('Посмотреть описание', callback_data='description')
              btn2 = telebot.types.InlineKeyboardButton('Купить', web_app=WebAppInfo(url = 'https://e-dracheva.github.io/'))
              markup.add(btn1, btn2)
              bot.send_photo(message.chat.id, 
                  img, 
                  caption = 
                  f'''
                  {search_result_dict[game]}
                  ''',
                  parse_mode='html',
                  reply_markup = 	markup)
        
    except:
        bot.send_message(message.chat.id, 'Извини, либо нет такой игры либо что-то пошло не так🙂', parse_mode='html')
        

@bot.message_handler(commands=['item_recs'])
def item_recs(message):
    msg = bot.send_message(message.chat.id, 'Помогу найти похожую игру. Введи название 🙂 (работает только на английском языке)', parse_mode='html')
    bot.register_next_step_handler(msg, item)
def item(message):
    try:
        title = b_games.loc[b_games['title'].str.contains(message.text, case = False)]['title'].values[0]
        get_similar_games(title, model, message)
    except:
        bot.send_message(message.chat.id, 'Извини, либо нет такой игры либо что-то пошло не так🙂', parse_mode='html')


@bot.message_handler(commands=['user_recs'])
def user_recs(message):
   
    conn = sqlite3.connect('database/user_data.sql')
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS user_info (id int auto_increment primary key, nickname varchar(50), pass varchar(50))')
    conn.commit()
    cur.close()
    conn.close()
    
    msg = bot.send_message(message.chat.id, 
                           'Чтобы дать персональные рекомендации, давайте поймем кто вы 🙂 \nВведите никнейм', 
                           parse_mode='html')
    bot.register_next_step_handler(msg, user_name)

def user_name(message):
    global nickname 
    nickname = message.text.strip()
    conn = sqlite3.connect('database/user_data.sql')
    cur = conn.cursor()
    result = cur.execute(f'SELECT nickname FROM user_info WHERE nickname="{nickname}"').fetchone()
    cur.close()
    conn.close()
    
    if result != None:
        msg = bot.reply_to(message, 'Отлично, такой пользователь существует, введите пароль 🙂', parse_mode='html')
        bot.register_next_step_handler(msg, user_pass)
    
    else:
        bot.reply_to(message, 'Такого пользователя не существует 🙂', parse_mode='html')
    
def user_pass(message):
    global nickname
    password = message.text.strip()
    conn = sqlite3.connect('database/user_data.sql')
    cur = conn.cursor()
    result = cur.execute(f'SELECT pass FROM user_info WHERE pass="{password}"').fetchone()
    cur.close()
    conn.close()
    if result != None:
        bot.reply_to(message, f'Привет, {nickname}! Давай я покажу тебе твои персональные рекомендации 🙂', parse_mode='html')
        generate_personal_recs(message, nickname)
    else: 
        bot.reply_to(message, 'Неправильный пароль, попробуй еще раз 🙂', parse_mode='html')
        

def user_name_new(message):
    global nickname 
    nickname = message.text.strip()
    bot.send_message(message.chat.id, 'Введите пароль 🙂', parse_mode='html')
    bot.register_next_step_handler(message, user_pass_new)
    
def user_pass_new(message):
    password = message.text
    bot.send_message(message.chat.id, 'Введите пароль 🙂', parse_mode='html')
    bot.register_next_step_handler(message, user_pass_new)

    conn = sqlite3.connect('database/user_data.sql')
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
    
    if call.data=='users':
        conn = sqlite3.connect('database/user_data.sql')
        cur = conn.cursor()
      
        cur.execute("SELECT * FROM user_info")
        users = cur.fetchall()
      
        info = ''
        for el in users:
            info += f'Имя: {el[1]}\n'
      
        cur.close()
        conn.close()
        bot.send_message(call.message.chat.id, info)
    
    if call.data=='category':
        bot.send_message(call.message.chat.id, get_categories)
        
    if call.data == 'description':
            game = b_games[b_games['title'].str.contains(pat = call.message.caption, case = False)].head(1)
            clear_description = BeautifulSoup(game['description'].values[0], features="lxml").get_text()
            bot.send_message(call.message.chat.id, 
                    text = f'''
<b>Название</b>: {game.title.values[0]}
<b>Рейтинг</b>: {game.average_rating.values[0].astype(float).round(2)}
<b>Минимальное количество игроков</b>: {game.minplayers.values[0]}
<b>Максимальное количество игроков</b>: {game.maxplayers.values[0]}
<b>Время игры</b>: {game.maxplaytime.values[0]} минут 
<b>Минимальный возраст для игры</b>: {game.age.values[0]}
<b>Механики</b>: {game.mechanics.values[0].replace('|', ', ')}
<b>Категория</b>: {game.category.values[0].replace('|', ', ')}
<b>Описание</b>: {clear_description}
                    ''',
                    parse_mode='html'
                    )
            
          

bot.polling(non_stop=True)

















