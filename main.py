from tools.rec_sys import Recommender
from tools.cf_rec import CFRecommender
import argparse
import sys

def parse_args():

    parser = argparse.ArgumentParser(description='rec sys')

    parser.add_argument(
        "--process",
        "-p",
        action="store_true"
    )

    parser.add_argument(
        "--user_id",
        "-uid",
        type=str,
        default='02a4255f067037ab82375a12d941a3df6ba93248'
    )

    parser.add_argument(
        "--song_id",
        "-sid",
        type=str,
        default='SOAUWYT12A81C206F1'
    )

    args = parser.parse_args()
    return args.process, args.user_id, args.song_id


def main():

    process, user_id, song_id = parse_args()

    recommender = Recommender()

    themes = ['love', 'war','happiness']
    # genres = ['Rock', 'Rap', 'Jazz', 'Electronic', 'Pop', 'Blues', 'Country', 'Reggae', 'New Age']

    df = recommender.top_250_tracks()

    print("Top-250 Tracks")
    print(df.head(5))
    print(df.tail(5))

    print('-' * 20)

    print("Top-100 tracks by genre")

    df = recommender.top_100_tracks_by_genre('Rock')

    print("Top 100 of genre 'Rock'")
    print(df.head(5))
    print(df.tail(5))

    df = recommender.top_100_tracks_by_genre('Rap')
    print("Top 100 of genre 'Rap'")
    print(df.head(5))
    print(df.tail(5))

    df = recommender.top_100_tracks_by_genre('Electronic')
    print("Top 100 of genre 'Electronic'")
    print(df.head(5))
    print(df.tail(5))

    print('-' * 20)
    print("Collections")

    print("baseline approach")
    df = recommender.collections('love')
    print(df.shape)
    print(df.head(10))

    print("word2vec approach")
    for theme in themes:

        df = recommender.collections(theme, approach='word2vec')
        print(df.shape)
        print(df.head(10))
    
    print("classification approach")
    df = recommender.collections('war', process=process, approach='classificaiton')
    print(df.shape)
    print(df.head(10))

    print('-' * 20)
    
    del recommender, df

    print("Collaborative filtering approach")

    cf_rec = CFRecommender()

    df = cf_rec.load_data()

    train_data, test_data = cf_rec.load_split()

    user_item_matrix = cf_rec.load_sparse()

    cf_rec.load_mapping()

    model = cf_rec.load_model()

    song_df = cf_rec.load_data(True)

    print("dataPeople similar to you listen")
    rec_df, precision = cf_rec.similar_to_you_listen(
        user_id=user_id,
        song_df=song_df,
        test_data=test_data,
        model=model,
        user_item_matrix=user_item_matrix
    )

    print(f"precision at 10 for this recommendation: {(precision * 100):.2f} %")
    print(rec_df)

    print("-" * 20)

    print("dataPeople who listen to this track usually listen")
    rec_df = cf_rec.get_similar_songs(
        song_id=song_id,
        train_data=train_data,
        song_df=song_df,
        model=model
    )

    user_actual = set(test_data[test_data.user_id == user_id]['song_id'])
    precision = cf_rec.calulate_precision(rec_df['song_id'], user_actual, 10)

    print(f"precision of this 10 recommendations: {(precision*100):.2f}")
    print(rec_df[['artist', 'title', 'likelihood']])

    del model, song_df, train_data, test_data, user_item_matrix

    print("-" * 20)
    print("Bonus")
    recommender = Recommender()
    print("Top listened song based on user history")
    df = recommender.your_top_k_songs(user_id)
    print(df)

    print("Top songs of favorite artist")

    df = recommender.top_songs_from_favorite_artist(user_id)
    print(df)

    print("Top songs of most genre played")
    df = recommender.song_genre_based(user_id)
    print(df)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"exception : {str(e)}")
        sys.exit(1)