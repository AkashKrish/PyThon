'''Docstring'''
#pylint: disable=C0103
import sqlite3
import sys

from tweepy import OAuthHandler, API


def create_cursor(data):
    '''Create connection with SQLite Data Base'''
    conn = sqlite3.connect(data)
    c = conn.cursor()
    return c

def create_table(db):
    '''Create a table in the database'''
    db.execute('''CREATE TABLE MyFeed IF NOT EXISTS(
                    Time    DATETIME,
                    UserID  TEXT,
                    Message TEXT 
                    )'''
              )
    db.commit()
    return True

def insert_data(db, data):
    '''Insert the data into DataBase'''
    db.execute('''
                    INSERT INTO MyFeed (Time, UserID, Message)
                    VALUES(?, ?, ?)
               ''', (data)
              )
    db.commit()
    return True

def delete_data(db):
    '''Insert the data into DataBase'''
    db.execute('''
                   DELETE FROM MyFeed
               ''',
              )
    db.commit()
    return True

def Oauth_Twitter():
    '''Initialize connection to twitter'''
    #consumer key, consumer secret, access token, access secret.
    ckey = "FUByaHr3JvidjsT7HA3Gt4CcH"
    csecret = "hNLZL0sXEah5M4VdbA3aVmuiIvD66I4IXg0kOLCC8I9IoF0p3l"
    atoken = "268097812-62SHOW9hAvKFG6T25bPTbHt3g10qQjsQBECPxAV7"
    asecret = "bFp3MncAYsExCTlsDxRQI8Od51w7Br5LBTDnPpnMKK46K"

    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)
    tweepy_api = API(auth)
    return tweepy_api

def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)

def main():
    '''Main Function for testing tweepy'''
    data_path = 'C:\\users\\Akash\\Documents\\Github\\Data Sets\\Tweet\\Tweet.DB'
    connection = create_cursor(data_path)

    api = Oauth_Twitter()
    public_tweets = api.get_user('twitter')

    uprint(api.home_timeline()[0].user)


if __name__ == '__main__':
    sys.exit(int(main() or 0))
