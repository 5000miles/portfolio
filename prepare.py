import pandas as pd
from nltk.tokenize import RegexpTokenizer,sent_tokenize
from nltk.stem import WordNetLemmatizer

def prepare(data_df):
    # combine title column and selftext column
    # check if the selftext is null and create a new column based on the result
    data_df['selftext_null'] = data_df['selftext'].isnull().astype(int)
    # using apply function to combine, if selftext is null then return title, otherwise, combine them.
    data_df['title_selftext'] = data_df.apply(lambda x: x['title'] if (x['selftext_null']==1) else (x['title']+x['selftext']),axis=1)

    # tokenize the title_selftext
    regex_token = RegexpTokenizer('\w+')
    data_df['title_selftext_token'] = data_df['title_selftext'].map(lambda x: regex_token.tokenize(x.lower()))

    # lemmitize the title_selftext_token
    lemmatizer = WordNetLemmatizer()
    data_df['title_selftext_token'] = data_df['title_selftext_token'].map(lambda x:[lemmatizer.lemmatize(i) for i in x])

    # convert list back to string in order to CountVectorizer
    data_df['title_selftext_new'] = data_df['title_selftext_token'].map(lambda x: ','.join(x))

    return data_df

def input_to_df(title_text, self_text):
    result_df = pd.DataFrame({'selftext':[self_text], 'title':[title_text]})

    return result_df
