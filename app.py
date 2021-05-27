import streamlit as st
import pickle
import numpy as np
import base64


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return




page = st.sidebar.selectbox(
'Navigator:',
(
    'About Me',
    'NLP project',
    'Make a prediction')
)

if page == 'About Me':

    set_png_as_page_bg('L1010977.jpg')

    st.markdown("""
    <style>
    .big-font {
        font-size:95px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Hello Welcome!</p>', unsafe_allow_html=True)

    st.code("class Me: \n\
     def __init__(self, name, background, skills): \n\
        self.name = 'Trevor' \n\
        self.background = 'A Data Scientist/Analyst with Math and Statistics background' \n\
        self.skills = 'Python is my first language, but I am also good at R.'\n\
        \n\
    class MoreMe:")



if page == 'NLP project':
    st.image('NLP/reddit.png')
    st.title('Which subreddit is the post from?')
    st.markdown('''
    ## Web APIs & NLP

    ### Description
    I collected text data from 2 subreddits: `r/math` and `r/Physics`,
    then utlizing multiple classifers to classify posts, predicting which subreddit it comes \
    from based on its title and self text. [You can find my full project here]("https://git.gehttps://github.com/5000miles/models_practice_review/tree/main/Projects/project_3-master")

    #### API

    > API: [Pushshift's API ]('https://pushshift.io/')


    #### Data Source

    Subreddits:
    > * [r/math]('https://www.reddit.com/r/math/'): 1.5m members, Created Jan 24, 2008
    > * [r/Physics]('https://www.reddit.com/r/Physics/'): 1.7m members, Created Mar 16, 2008

    ---

    ### Data info

    - 5000 observations from 11/24/2020 to 3/29/2021 for each subreddit, excluded `[removed]` and `[deleted]` posts.
        * `[removed]` posts are results of automod action, because those posts triggered a filter from reddit's website.
        * `[deleted]` posts are posts that deleted by the author self.


    - The combination of `tile` & `selftext` is the predictor, `math`/`Physics` is the response variable.

    ---

    ### EDA/ Preparing the data

    - Created distribuition plots for the posts in each subreddit. Clearly see that redditors like to post at around noon for both subreddit.
    '''
    )
    st.image('NLP/distribution.jpg')

    st.markdown(
    '''
    - Using `RegexpTokenizer` to tokenize the predictors with Regex: `\w+`
    - Using `WordNetLemmatizer` to lemmatize the predicotrs
    - Combine 2 dateframes(math & physics)

    ### Model building:


    Built 6 models with `GridSearchCV`:

    | Model | Vectorize |
    | --- | --- |
    | **Logistic Regression** | *Using TfidfVectorizer()* |
    | **ordinary Decision Tree** | *Using TfidfVectorizer()* |
    | **Random Forest** | *Using TfidfVectorizer()* |
    | **Adaboost** | *Using TfidfVectorizer()* |
    | **SVM** | *Using TfidfVectorizer()* |
    | **Naive Bayes** | *Using TfidfVectorizer()* |



    ### Model results:


    | Model | Precision | Recall |
    | --- | --- | --- |
    | **Logistic Regression** | *0.866564* | *0.9040* |
    | **SVM** | *0.858315* | *0.9208* |
    | **ordinary Decision Tree** | *0.779221* | *0.8640* |
    | **Random Forest** | *0.832969* | *0.9136* |
    | **Adaboost** | *0.793033* | *0.9288* |
    | **Naive Bayes** | *0.834418* | *0.9232* |




    ### Using three models to make prediction on data in different time period.

    100 observations randomly chosen from 3/1/2015 - 9/1/2015, this is out of the time range of the training and testing data.


    | Model | Precision | Recall |
    | --- | --- | --- |
    | **Logistic Regression** | *0.923077* | *0.96* |
    | **Random Forest** | *0.834783* | *0.96* |
    | **Adaboost** | *0.770492* | *0.94* |


    ### Conclusion

    The models are doing pretty good.
    Logistic Regression and SVM have the best performance.
    Logistic Regression has a slightly higher precision than SVM, but SVM's recall is higher.
    ''')

    st.title('Try my model!')


    st.write('\n Input a title and selftext below and my model will tell you which subreddit it comes from.')
    model = pickle.load(open('project3.pkl','rb'))
    # context = st.text_input("Enter the title of the Reddit post","")
    #     if st.button('RUN!'):
    #         if len(context) == 0

if page == 'Make a prediction':
    model = pickle.load(open('project3.pkl','rb'))
    title = st.text_input("Enter the title of the Reddit post","")


    if title:
        predictor = np.array([title])
        st.write(f'The title you entered is a Reddit Post of')
        st.write(f'{model.predict(predictor)[0]}')
        st.write('subreddit')
