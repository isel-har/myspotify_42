import os

os.environ['GENSIM_DATA_DIR'] = '/Users/isel-har/goinfre/gensim'

import numpy as np
import pandas as pd
import duckdb
import joblib
import nltk

import gensim.downloader as api
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from scipy.sparse import lil_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from implicit.als import AlternatingLeastSquares


pd.set_option('display.max_columns', None)

class Recommender:

    stop_words = set(stopwords.words("english")) 

    def __init__(self):

        self.collection_pass = False
        self.is_loaded = False
        self.clf = None
        self.classes_ = None
        self.track_ids = None
        self.y_pred = None
        self.pred_db = None
        self.w2v_model = None
        self.processed = False
        self.user_encoder = None

        self.themes = {
            'love': ['love', 'heart', 'kiss', 'romance', 'lover', 'baby', 'honey'],
            'war': ['war', 'fight', 'battle', 'soldier', 'gun', 'blood'],
            'happiness': ['happy', 'joy', 'smile', 'laugh', 'celebration', 'party'],
            'loneliness': ['lonely', 'alone', 'sad', 'cry', 'empty', 'miss'],
            'money': ['money', 'rich', 'dollar', 'gold', 'cash', 'wealth']
        }

        self.mds_tagtraum_db = """read_csv('data/p02_msd_tagtraum_cd2.cls', comment='#', columns={
                    'track_id':'VARCHAR',
                    'genre':'VARCHAR'
                },
                ignore_errors=true,
                delim='\t'
        )"""


        self.unique_tracks = """read_csv('data/p02_unique_tracks.txt',
                delim='\n',
                header = false,
                columns={'line':'VARCHAR'}
        """

        self.unique_tracks_db = f"""SELECT string_split(line, '<SEP>') AS parts
                         FROM {self.unique_tracks}"""


        """
            Correct and cleaned datasets!
        """
    
        self.train_triplets_db = """read_csv('data/train_triplets.txt', delim='\t', header = false,
                    columns={'user_id':'VARCHAR', 'song_id':'VARCHAR', 'play_count':'INTEGER'}
        )"""

        self.p02_unique_tracks_db = pd.read_csv("data/p02_unique_tracks.txt",
            header=None,
            names=['track_id', 'song_id', 'artist', 'title'],
            delimiter='<SEP>',
            engine='python'
        )
        self.p02_msd_tagtraum_cd2 = pd.read_csv("data/p02_msd_tagtraum_cd2.cls",
            header=None,
            comment='#',
            names=['track_id', 'majority', 'minority'],
            delimiter='\t',
            engine='python'
        )

        self.p02_msd_tagtraum_cd2.drop_duplicates(subset='track_id', inplace=True)
        self.p02_unique_tracks_db.drop_duplicates(subset='song_id', inplace=True)
        

    def load_unique_tracks_db(self):
        self.p02_unique_tracks_db = pd.read_csv("data/p02_unique_tracks.txt",
            header=None,
            names=['track_id', 'song_id', 'artist', 'title'],
            delimiter='<SEP>',
            engine='python'
        )
        self.p02_unique_tracks_db.drop_duplicates(subset='song_id', inplace=True)

        return self.p02_unique_tracks_db


    def collection_filter_query(self, theme, db, track_id=False):
        return f"""
            SELECT
                {'tt.track_id,' if track_id else ''}
                tt.artist,
                tt.title,
                SUM(tt.play_count) AS play_count
            FROM
                ({self.triplets_tracks_db('tdb.artist, tdb.title, sdb.play_count, tdb.track_id')}) AS tt
            JOIN (select * from {db} where theme like '{theme}') as tm
                ON tt.track_id = tm.track_id
            GROUP BY    
                tt.track_id, tt.artist, tt.title

            ORDER BY play_count DESC
            LIMIT 50
        """

    def triplets_tracks_db(self, columns):

        return f"""
            SELECT
            {columns}
            FROM
            {self.train_triplets_db} as sdb
            JOIN
            p02_unique_tracks_db AS tdb
            ON sdb.song_id = tdb.song_id
        """


    def top_250_tracks(self):

        p02_unique_tracks_db = self.p02_unique_tracks_db

        query = f"""
            select p.artist, p.title, sum(t.play_count) as play_count  from {self.train_triplets_db} as t
            join p02_unique_tracks_db as p on t.song_id = p.song_id
            group by p.song_id, p.artist, p.title
            order by play_count desc
            limit 250
        """
        result = duckdb.query(query).to_df()

        return result


    def get_genres(self):

        return duckdb.query(f"""
            select distinct genre
            from {self.mds_tagtraum_db}
        """).to_df()['genre'].tolist()


    def top_100_tracks_by_genre(self, genre):

        p02_msd_tagtraum_cd2 = self.p02_msd_tagtraum_cd2
        p02_unique_tracks_db = self.p02_unique_tracks_db

        cte = f"""
            WITH tracks_table AS (
                SELECT 
                    p.track_id,
                    p.artist,
                    p.title,
                    sum(t.play_count) as play_count
                FROM {self.train_triplets_db} AS t
                JOIN p02_unique_tracks_db AS p
                ON t.song_id = p.song_id
                GROUP BY p.track_id, p.artist, p.title
            )
        """

        base_query = f"""
            {cte}
            SELECT tt.artist, tt.title, tt.play_count
            FROM tracks_table AS tt
            JOIN p02_msd_tagtraum_cd2 as mds
            ON tt.track_id = mds.track_id
        """

        result =  duckdb.query(f"""
            {base_query}
            WHERE mds.majority like '{genre}'
            ORDER BY tt.play_count DESC
            limit 100
        """)
        return result.to_df()



    def word_vec(self, theme,top_n=10):
    
        if theme in self.w2v_model:
            return self.w2v_model.most_similar(theme, topn=top_n)
        return None


    def collection(self, theme, threshold=0.1, word2vec=False, top_n=10, min_theme_words=5):

        if theme not in self.themes:
            print(f"Incorrect given keyword '{theme}'")
            return pd.DataFrame()


        collection = {'track_id': [], 'theme':[], 'theme_ratio':[]}

        theme_index = set()

        for val in self.themes.get(theme, []):
            idx = self.keyword_map.get(val)
            if idx is not None:
                theme_index.add(idx)
            
        if word2vec:
            if self.w2v_model is None:
                self.w2v_model = api.load("word2vec-google-news-300")

            similar_tokens = self.word_vec(theme=theme, top_n=top_n)

            for token, score in similar_tokens:
                idx = self.keyword_map.get(token)
                if idx is not None:
                    theme_index.add(idx)
        
        with open('data/mxm_dataset_train.txt', 'r') as f:

            for line in f:

                if line.startswith(('%', '#')):
                    continue

                parts = line.strip().split(',')
                track_id = parts[0]

                theme_score = 0
                total_words = 0

                for part in parts[2:]:

                    word_index, count = part.split(':')
                    word_index        = int(word_index)
                    count             = int(count)

                    total_words += count

                    if word_index in theme_index:
                        theme_score += count

                if total_words == 0:
                    continue
                
                theme_ratio = theme_score / total_words

                if theme_ratio > threshold and theme_score >= min_theme_words:
                        collection['track_id'].append(track_id)
                        collection['theme_ratio'].append(theme_ratio)


        collection['theme'] = theme
        data_frame = pd.DataFrame(collection)

        if data_frame.empty:
            print("No tracks found for theme:", theme)
            return pd.DataFrame()

        theme_db = duckdb.from_df(data_frame)

        p02_unique_tracks_db = self.p02_unique_tracks_db

        result   = duckdb.query(f"""
            SELECT 
                tt.artist,
                tt.title,
                SUM(tt.play_count) AS play_count
            FROM
                ({self.triplets_tracks_db('tdb.artist, tdb.title, sdb.play_count, tdb.track_id')}) AS tt
            JOIN theme_db tm
                ON tt.track_id = tm.track_id
            GROUP BY 
                tt.track_id, tt.artist, tt.title
            ORDER BY play_count DESC
            LIMIT 50
        """)

        print(data_frame.shape)
        if word2vec:
            data_frame.to_csv(f'data/{theme}_data.csv', index=False, header=False)
            print(f"data frame saved as data/{theme}_data.csv for training")
        return result.to_df()


    def mxm_dict(self):

        m_dict = {}

        with open('data/mxm_dataset_train.txt', 'r') as f:
            
            for line in f:

                words_count =  np.zeros(self.vocabulary_size)       
                if line.startswith(('%', '#')):
                    continue

                parts    = line.strip().split(',')
                track_id = parts[0]
                for part in parts[2:]:

                    word_index, count     = part.split(':')
                    word_index            = int(word_index)
                    count                 = int(count)
                    if word_index not in self.stop_words_idx:
                        words_count[word_index - 1] = count
                
                m_dict[track_id] = words_count

        return m_dict


    def vectorizer(self, mxm_dict=None, track_id_list=None):
        
        vectors_list = []
        for track_id in track_id_list:
            try:
                vectors_list.append(mxm_dict[track_id])
            except Exception as e:
                print("track id :", track_id)
                print("error :", str(e))
        return np.array(vectors_list)  


    def preprocessing(self):
        
        df = None
        for theme in ['love', 'war', 'happiness']:
            theme_df = pd.read_csv(
                f"data/{theme}_data.csv",
                names=['track_id', 'theme', 'theme_ratio'],
                header=None
            )
            df = pd.concat([df, theme_df])

        df = df.sample(frac=1)
        df = df.drop_duplicates()

        X_train, X_test, y_train, y_test = train_test_split(
            df['track_id'],
            df['theme'],
            test_size=0.2,
            stratify=df['theme'],
            random_state=42
        )

        le   = LabelEncoder()
        mx_dict = self.mxm_dict()

        joblib.dump(X_test, "data/track_ids_test.pkl")
        # joblib.dump(X_train, "data/track_ids_train.pkl")
        # joblib.dump(y_train, "data/track_themes.pkl")

        X_train = self.vectorizer(mx_dict, X_train.values.tolist()).astype(np.float32)
        X_test  = self.vectorizer(mx_dict, X_test.values.tolist()).astype(np.float32)


        le.fit(pd.concat([y_train, y_test]))
        y_train  = le.transform(y_train)
        y_test  =  le.transform(y_test)
    
        joblib.dump(X_train, "data/X_train.pkl")
        joblib.dump(X_test, "data/X_test.pkl")
        joblib.dump(y_train, "data/y_train.pkl")
        joblib.dump(y_test, "data/y_test.pkl")
        joblib.dump(le.classes_, "data/classes.pkl")

        print("preprocessed train/test split and classes saved at data/")


    def classifier(self):
        
        X_train, y_train, X_test, y_test = joblib.load('data/X_train.pkl'), joblib.load('data/y_train.pkl'), \
            joblib.load('data/X_test.pkl'), joblib.load('data/y_test.pkl')
    
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            batch_size=16,
            random_state=42,
            max_iter=100,
            solver='adam',
            activation='relu',
            early_stopping=True
        )

        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        clf.fit(X_train, y_train, sample_weight=sample_weight)

        self.y_pred = clf.predict(X_test)
        print(f"classifier accuracy reached on test set: {accuracy_score(y_pred=self.y_pred, y_true=y_test)}")
        return clf


    def collection_classification(self, theme):

        if not self.is_loaded:
            self.classes_, self.track_ids = joblib.load('data/classes.pkl'), joblib.load('data/track_ids_test.pkl')
            self.clf = self.classifier()

            themes = [self.classes_[l] for l in self.y_pred.tolist()]
            self.pred_db = duckdb.from_df(
                pd.DataFrame({
                    'track_id': self.track_ids,
                    'theme': themes
                })
            )

            self.is_loaded = True

        pred_db = self.pred_db
        p02_unique_tracks_db = self.p02_unique_tracks_db
        query = self.collection_filter_query(theme, "pred_db", track_id=True)


        result = duckdb.query(query).to_df()

        # return result
        rows = len(result)
        target_rows = 50

        if rows < target_rows:
            needed = target_rows - rows

            original_df = pd.read_csv(f"data/{theme}_data.csv", header=None, names=['track_id', 'theme', 'ratio'])

            merged = original_df.merge(
                result, how='left', indicator=True, on='track_id'
            )
            not_in_query_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

            if len(not_in_query_df) > 0:
                not_in_query = duckdb.from_df(not_in_query_df)
                padding_query = self.collection_filter_query(theme, "not_in_query")
                padding_result = duckdb.query(padding_query).to_df().head(needed)

                result = pd.concat([result, padding_result], ignore_index=True)

        return result.head(target_rows)[['artist', 'title', 'play_count']]

        
    def collections(self, theme, process=False, approach='baseline'):
        
        if not self.collection_pass:
            keywords = pd.read_csv('data/mxm_dataset_train.txt', comment='#', nrows=1) \
                .columns.to_list()
            keywords[0] = 'i'
            self.keyword_map = {
                w: i for i, w in enumerate(keywords, start=1)
                if w not in self.stop_words 
            }
            self.vocabulary_size = max( self.keyword_map.values() )
            self.stop_words_idx = {
                w: i for i, w in enumerate(keywords, start=1)
                if w in self.stop_words
            }

            self.collection_pass = True

        print(f"theme : {theme}")
        if approach in ('baseline', 'word2vec'):

            is_wv = approach == 'word2vec'
            return self.collection(
                theme,
                threshold=0.063,
                word2vec=is_wv,
                top_n=10,
                min_theme_words=4
            )
        
        self.w2v_model = None

        if process:
            if not self.processed:
                print("data processing...")
                self.preprocessing()
                self.processed = True
        
        return self.collection_classification(theme)


    ##=========================Bonus elements==============================##

    def your_top_k_songs(self, user_id, n=5):
        p02_unique_tracks_db = self.p02_unique_tracks_db
        query = f"""
            SELECT
                p.song_id,
                p.artist,
                p.title,
                SUM(t.play_count) AS total_plays
            FROM {self.train_triplets_db} t
            JOIN p02_unique_tracks_db p
                ON t.song_id = p.song_id
            WHERE t.user_id = '{user_id}'
            GROUP BY
                p.song_id,
                p.artist,
                p.title
            ORDER BY total_plays DESC
            LIMIT {n}
        """

        return duckdb.query(query).to_df()




    def top_songs_from_favorite_artist(self, user_id, k=10):
        p02_unique_tracks_db = self.p02_unique_tracks_db
        # 1. Find user's top artist
        top_artist_query = f"""
            SELECT
                p.artist,
                SUM(t.play_count) AS total_plays
            FROM {self.train_triplets_db} t
            JOIN p02_unique_tracks_db p
                ON t.song_id = p.song_id
            WHERE t.user_id = '{user_id}'
            GROUP BY p.artist
            ORDER BY total_plays DESC
            LIMIT 1
        """

        top_artist_df = duckdb.query(top_artist_query).to_df()

        if top_artist_df.empty:
            return top_artist_df

        artist = top_artist_df["artist"][0]

        # 2. Get top songs from that artist
        songs_query = f"""
            SELECT
                p.song_id,
                p.artist,
                p.title,
                SUM(t.play_count) AS score
            FROM {self.train_triplets_db} t
            JOIN p02_unique_tracks_db p
                ON t.song_id = p.song_id
            WHERE p.artist = '{artist}'
            GROUP BY p.song_id, p.artist, p.title
            ORDER BY score DESC
            LIMIT {k}
        """

        return duckdb.query(songs_query).to_df()
    

    def song_genre_based(self, user_id, n=5):
        p02_unique_tracks_db = self.p02_unique_tracks_db
        p02_msd_tagtraum_cd2 = self.p02_msd_tagtraum_cd2
    
        genre_query = f"""
            SELECT
                m.majority,
                SUM(t.play_count) AS total_plays
            FROM {self.train_triplets_db} t
            JOIN p02_unique_tracks_db p
                ON t.song_id = p.song_id
            JOIN p02_msd_tagtraum_cd2 m
                ON p.track_id = m.track_id
            WHERE t.user_id = '{user_id}'
            GROUP BY m.majority
            ORDER BY total_plays DESC
            LIMIT 1
        """

        genre = duckdb.query(genre_query).fetchone()[0]

        query = f"""
            SELECT
                p.song_id,
                p.artist,
                p.title,
                m.majority AS genre,
                SUM(t.play_count) AS popularity
            FROM p02_unique_tracks_db p
            JOIN p02_msd_tagtraum_cd2 m
                ON p.track_id = m.track_id
            JOIN {self.train_triplets_db} t
                ON p.song_id = t.song_id
            WHERE m.majority = '{genre}'
            AND p.song_id NOT IN (
                SELECT song_id
                FROM {self.train_triplets_db}
                WHERE user_id = '{user_id}'
            )
            GROUP BY
                p.song_id,
                p.artist,
                p.title,
                m.majority
            ORDER BY popularity DESC
            LIMIT {n}
        """

        return duckdb.query(query).to_df()


