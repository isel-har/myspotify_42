from implicit.als import AlternatingLeastSquares
from scipy.sparse import  save_npz, load_npz
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
import joblib



class CFRecommender:
    def __init__(self):

        self.user_to_idx = None
        self.song_to_idx = None
        self.idx_to_user = None
        self.idx_to_song = None


    def load_data(self, song_df=False):

        data = None
        if not song_df:
            data = pd.read_csv(
                "data/train_triplets.txt",
                header=None,
                names=['user_id', 'song_id', 'play_count'],
                delimiter='\t'
            )
            data.dropna(inplace=True)
        else:
            data = pd.read_csv("data/p02_unique_tracks.txt",
                header=None,
                names=['track_id', 'song_id', 'artist', 'title'],
                delimiter='<SEP>',
                engine='python'
            )
            data.drop_duplicates(subset='song_id', inplace=True)
        return data



    def load_model(self, path='als_model.pkl'):
        model = joblib.load(path)
        if model is None:
            raise Exception("model is None")
        return model


    def load_split(self, path="./"):

        train = pd.read_csv(f"{path}train.csv")
        test = pd.read_csv(f"{path}test.csv")

        return train, test


    def load_mapping(self, path='./'):

        self.user_to_idx = joblib.load(f"{path}user_to_idx.pkl")
        self.idx_to_user = joblib.load(f"{path}idx_to_user.pkl")
        self.song_to_idx = joblib.load(f"{path}song_to_idx.pkl")
        self.idx_to_song = joblib.load(f"{path}idx_to_song.pkl")


    def load_sparse(self, path='data/'):
        return load_npz(f'{path}sparse_matrix.npz')



    def filter_data(
        self,
        data,
        play_count_threshold=50,
        song_threshold=125,
    ):
    
        full_data = data

        user_play_counts = full_data.groupby('user_id')['play_count'].sum()
    
        active_users = user_play_counts[
            user_play_counts >= play_count_threshold
        ].index
    
        filtered_data = full_data[full_data['user_id'].isin(active_users)]
        
        song_listener_counts = filtered_data.groupby('song_id')['user_id'].nunique()

        popular_songs = song_listener_counts[song_listener_counts >= song_threshold].index

        filtered_data = filtered_data[filtered_data['song_id'].isin(popular_songs)]

        print(filtered_data.shape)
        print(f"Users after filtering: {len(active_users)}")
        print(f"Songs after filtering: {len(popular_songs)}")

        return filtered_data




    def split_per_user(self, data=None, test_frac=0.2, seed=42, save_split=True):

        if data is None:
            return

        np.random.seed(seed)

        # Assign a random number to each row
        data = data.copy()
        print(f'data shape : {data.shape}')
        data['rand'] = np.random.rand(len(data))
        
        # Mark test rows per user
        data['is_test'] = data.groupby('user_id')['rand'].transform(
            lambda x: x <= test_frac if len(x) > 1 else False
        )

        train_data = data[~data['is_test']].drop(columns=['rand', 'is_test'])
        test_data = data[data['is_test']].drop(columns=['rand', 'is_test'])

        print(f'train data shape : {train_data.shape}')
        print(f'test data shape : {test_data.shape}')  
        if save_split:
            train_data.to_csv("train.csv", index=False)
            test_data.to_csv("test.csv", index=False)

        return train_data, test_data




    def create_sparse_matrix(
        self,
        train_data,
        save_sparse=True,
        save_mapping=True
    ):
    
        users = train_data['user_id'].unique()
        songs = train_data['song_id'].unique()

        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.song_to_idx = {song: idx for idx, song in enumerate(songs)}
        self.idx_to_user = {idx: user for idx, user in enumerate(users)}
        self.idx_to_song = {idx : song for idx, song in enumerate(songs)}

        rows = train_data['user_id'].map(self.user_to_idx)
        cols = train_data['song_id'].map(self.song_to_idx)
        values = train_data['play_count'].values

        user_item_matrix = coo_matrix(
            (values, (rows, cols)),
            shape=(len(users), len(songs))
        ).tocsr()


        if save_sparse: save_npz("data/sparse_matrix.npz", user_item_matrix)
        if save_mapping:
            joblib.dump(self.user_to_idx, "user_to_idx.pkl")
            joblib.dump(self.song_to_idx, "song_to_idx.pkl")
            joblib.dump(self.idx_to_user, "idx_to_user.pkl")
            joblib.dump(self.idx_to_song, "idx_to_song.pkl")
            print("user item mapping saved")
            


        return user_item_matrix


    def train(
        self,
        user_item_matrix,
        factors=128,
        iters=50,
        regularization=0.08,
        alpha=40,
        seed=42,
        save_model=True
    ):
        
        model = AlternatingLeastSquares(
            factors=factors,
            iterations=iters,
            regularization=regularization,
            alpha=alpha,
            random_state=seed
        )

        model.fit(user_item_matrix)

        if save_model:
            joblib.dump(model, "als_model.pkl")
            print("model saved")

        return model



    def calulate_precision(self, recommended_songs, actual_songs, K):
        hits = len(set(recommended_songs) & set(actual_songs))
        precision = hits / K
        return precision
    

    def recommend_and_precision(
        self,
        user_item_matrix,
        model,
        user_id,
        test_data, 
        user_to_idx,
        idx_to_song,
        return_df=False,
        K=10
    ):

        if user_id not in user_to_idx:
            raise ValueError("User not in mapping!")
        user_idx = user_to_idx[user_id]


        user_row = user_item_matrix[user_idx] 
        recommended = model.recommend(user_idx, user_row, N=K)
        recommended_indices, scores = recommended
    
        rec_pairs = sorted(
            zip(recommended_indices, scores),
            key=lambda x: (-x[1], x[0])
        )
    
        recommended_songs = [idx_to_song[idx] for idx, _ in rec_pairs]
        scores = [score for _, score in rec_pairs]

        actual_songs = test_data[test_data.user_id == user_id]['song_id'].tolist()

        # --- Calculate precision@K ---
        precision = self.calulate_precision(recommended_songs, actual_songs, K)

        rec_df = None
        if return_df:
            rec_df = pd.DataFrame({
                'song_id' : recommended_songs,
                'likelihood' : scores
            })
            rec_df.sort_values('likelihood', ascending=False).reset_index(drop=True, inplace=True)

        return  rec_df, precision



    def evaluate_model(
        self,
        model,
        test_data,
        user_item_matrix,
        n_users=20,
        k=10
    ):
    
        print(f"\n Evaluating on {n_users} test users...")
        test_users = list(test_data['user_id'].unique())[:n_users]
        
        precisions = []
        for test_user in test_users:
            if test_user in self.user_to_idx:
                try:
                    _, user_precision = self.recommend_and_precision(
                        user_item_matrix=user_item_matrix,
                        model=model,
                        user_id=test_user,
                        test_data=test_data,
                        user_to_idx=self.user_to_idx,
                        idx_to_song=self.idx_to_song,
                        K=k
                    )
                    precisions.append(user_precision)
                except Exception as e:
                    print("excpetion :", str(e))
                    continue


        if precisions:
            avg_precision = np.mean(precisions)
            print(f"Average Precision@10: {avg_precision:.2%} ({len(precisions)} users)")
        else:
            print("Could not evaluate any users")



    def similar_to_you_listen(
        self,
        user_id,
        test_data,
        user_item_matrix,
        song_df,
        model,
        k=10
    ):

        if user_id not in self.user_to_idx:
            print(f"User {user_id} not in training data")
            available_users = list(self.user_to_idx.keys())
            if available_users:
                user_id = available_users[0]
                print(f"Using user {user_id} instead")
            else:
                return


        rec_df, precision = self.recommend_and_precision(
            user_item_matrix=user_item_matrix,
            model=model,
            user_id=user_id,
            test_data=test_data,
            user_to_idx=self.user_to_idx,
            idx_to_song=self.idx_to_song,
            return_df=True,
            K=k
        )
    
        result_inner = pd.merge(rec_df, song_df, on='song_id')[['artist', 'title', 'likelihood']]

        return result_inner, precision 
        


    def get_similar_songs(
        self,
        song_id,
        train_data,
        song_df,
        model,
        K=10,
        merge_song_df=True
    ):

        if song_id not in self.song_to_idx:
            print("Track not in mapping!")
            song_id = 'SOWYSKH12AF72A303A'
            print(f'Track id that exist : {song_id}')
        
        song_idx = self.song_to_idx[song_id]
        similar_items, scores = model.similar_items(song_idx, N=20)
        sim_pairs = sorted(zip(similar_items, scores), key=lambda x: (-x[1], x[0]))
        similar_items, scores = zip(*sim_pairs)

        filtered_tracks = []
        filtered_scores = []
        for idx, score in zip(similar_items[1:], scores[1:]):
    
            candidate_id = self.idx_to_song[idx]
            if len(train_data[train_data.song_id == candidate_id]) >= 30:
                filtered_tracks.append(candidate_id)
                filtered_scores.append(score)
            if len(filtered_tracks) >= K:
                break


        print(f'Size of filtered_tracks: {len(filtered_tracks)}')

        recommended_df = pd.DataFrame({
            'song_id': list(filtered_tracks[:K]),
            'likelihood': list(filtered_scores[:K])
        })

        recommended_df = pd.merge(recommended_df, song_df, on='song_id')
        recommended_df.sort_values('likelihood', ascending=False, inplace=True)
        recommended_df.reset_index(drop=True, inplace=True)

        return recommended_df



    def average_precision_at_k_given_track(
        self,
        song_id,
        train_data,
        test_data,
        song_df,
        model,
        K=10
    ):

        recommended_df = self.get_similar_songs(
            song_id,
            train_data,
            song_df,
            model,
            K,
            False
        )
    
        recommended_songs = recommended_df['song_id'].tolist()

        test_users_all = test_data[test_data.song_id == song_id]['user_id'].tolist()

        # Prefer users with more test data
        user_activity = test_data.groupby('user_id').size()
        test_users = sorted(
            test_users_all,
            key=lambda x: user_activity.get(x, 0),
            reverse=True
        )[:100]

        precisions = []
        for user_id in test_users:

            user_actual = set(test_data[test_data.user_id == user_id]['song_id'])
            result = self.calulate_precision(recommended_songs, user_actual, K)
            precisions.append(result)

        avg_precision = np.mean(precisions) if precisions else 0.0
        print(f"Average Precision@{K}: {avg_precision:.2%}")

        return avg_precision
