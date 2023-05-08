# MyFirstDataProject
Проект для трека https://ods.ai/tracks/my_first_data_project

В данном проекте реализуется телеграм-бот, который рекомендует настольные игры и по желанию позволяет купить понравившуюся игру. Этот бот может помочь вам выбрать настольную игру, предлагая рекомендации на основе популярности, сходства с другой игрой или персональных предпочтений, если у вас есть учетная запись (добавление новых учетных записей пока не предусмотрено). Вы также можете запросить случайно выбранную игру. Важно отметить, что названия и описания игр только на английском языке.

Сам телеграм-бот - https://t.me/Board_Recommender_Bot 

Для обучения модели использовалась библиотека Implicit для построения рекомендательных систем на основе датасетов с неявным таргетом.

<b>Таблица сравнения используемых моделей:</b>
<img width="1008" alt="Снимок экрана 2023-04-18 в 17 21 03" src="https://user-images.githubusercontent.com/122459598/232806988-5602419b-c27f-430e-bdb6-e97ac98fb412.png">

nearest_neighbours (CosineRecommender, BM25Recommender, TFIDFRecommender) - метод ближайшего соседа – поиск похожих объектов (по косинусной близости) для всех объектов, с которыми пользователь уже взаимодействовал и выдача топа из этого списка.

ALS предсказывает не исходное значение взаимодействия, а предсказывает факт такого взимодействия. Для каждой пары пользователь-объект есть вес, даже для неизвестных пар.

По функционалу качества чуть лучше показала себя AlternatingLeastSquares(factors = 16, iterations = 30), ее и будем использовать для MVP.

Модели градиентного бустинга не использовались для этого проекта из-за нехватки времени, возможно, дополню позже.

<pre><code> Структура проекта:
├── database                                  # папка с базами данных <br>
│   ├── user_data.sql                         # база данных с данными пользователей для входа в УЗ
├── datasets                                  # папка с датасетами
│   ├── bgg_boardgames_top_2000.feather       # набор с данными о настольных играх
│   ├── bgg_ratings_top_2000.feather          # набор с данными о пользователях и их взаимодействиями с играми
├── models                                    # папка с моделями
│   ├── als_models.npz                        # модель на основе ALS, которую мы сохранили для использования в боте
├── notebooks                                 # папка c ноутбуками
│   ├── board_game_recommender_notebook.ipynb # ноутбук на котором обучали и тестировали модели, смотрели метрики
├── .gitignore                                # содержит файлы, которые не должны попасть в Git
├── Dockerfile                                # файл для поднятия сервиса в Docker 
├── README.md                                 # описание проекта
├── bot.py                                    # файл разработки бота
├── requirements.txt                          # файл с зависимостями
</code></pre>

<h3>Как запустить сервис через Docker:</h3>

1. Образ есть на Docker Hub, поэтому можно скопировать его себе
<pre><code>docker pull eidracheva/my_bot</code></pre>
2. Обратите внимание, что внутри есть переменные окружения, поэтому для запуска нужно указать токен телеграм-бота в переменную TOKEN
<pre><code>docker run --rm -e TOKEN=*YOUR_TOKEN HERE* eidracheva/my_bot</code></pre>


