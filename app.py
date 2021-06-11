import streamlit as st
import pickle
import numpy as np
import base64
from prepare import prepare, input_to_df

# enlarge the font
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
    'Text Classification on Reddit',
    'Speech and Pictures Transformation')
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
     ")

     st.write("Please take a look of my projects that I recently did, and I'll keep updating them")

if page == 'Text Classification on Reddit':
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

    First model I tried is `Logistic Regression`, Logistic Regression model is
    always the first model I utlize for classify problem. Because Logistic Regression
    is easy to interpret. The coef of each variable(words in this case) reprensents
    clearly the importance of each variable.

    Second model I utilized is a `ordinary Decision Tree`. A ordinary Decision Tree
    has many disadvantages so I anticipated there will problems like `overfitting`
    or `low performance`, and I'll apply variants of Decision Tree later based on
    the result. As I expected, the accuracy score on training dataset is **0.984**
    and on testing dataset is **0.801**, overfitting is the main problem.

    Since overfitting is the main problem now, utilizing some **ensemble methods** is.
    necessay, first ensemble method I'm going to try is a `Bagged Decision Tree`.
    More specifically, `Random Forest` is the first one comes in my mind, because
    when using Random Forest, at each splitting within each **boostrapping sample**,
    a random **subset** of the features are select. If one or few features (words in
    this case) are very strong preictors for the response variable, these words
    will be used in many/all of the bagged decision tree, causing them to become
    correlated and result in high variance and overfitting, by selecting a random
    subset of features at each split, it will **counter this correlation between
    base trees**. The result is much better, Random Forest model got a score of **0.98**
    on training dataset as well but the score for testing dataset increased from
    **0.80** to **0.86**. I also used GridSearchCV from sklearn tried to find
    the best hyperparameters.

    The second ensemble method I utilized is `AdaBoost`.Baaged Decision Tree
    (including Random Forest), words are treated independly, however, sentences
    (especially written by human) couldn't be independly, so boosting decision
    tree might has better performance, since boosting decision tree is **fitted
    sequentially** instead of in parallel. AdaBoost is the Boosting Decision Tree
    I tried, however, the disadvantages of AdaBoost is it fits slow and is more
    prone to overfitting. However, the performance didn't get any improved. Accuracy
    score for training dataset is 0.93 and 0.85 for testing dataset. Less overfitting
    but worse perforance.

    The last two models I built utilized SVM and Naive Bayes.

    The chart below is the result for all 6 models.

    ### Model results:


    | Model | Precision | Recall |
    | --- | --- | --- |
    | **Logistic Regression** | *0.866564* | *0.9040* |
    | **SVM** | *0.858315* | *0.9208* |
    | **ordinary Decision Tree** | *0.779221* | *0.8640* |
    | **Random Forest** | *0.832969* | *0.9136* |
    | **Adaboost** | *0.793033* | *0.9288* |
    | **Naive Bayes** | *0.834418* | *0.9232* |

    ### Conclusion

    Logistic Regression and SVM have the best performance overall. Logistic
    Regression has a slightly higher precision than SVM, but SVM's recall is
    higher. However, this performance is calculated based the the testing dataset,
    next thing I'm going to do is I'll randomly select some observations from Reddit,
    and test which model will have the best performance on those unseen data.


    ### Using three models to make prediction on data in different time period.

    100 observations randomly chosen from 3/1/2015 - 9/1/2015,
    this is out of the time range of the training and testing data.


    | Model | Precision | Recall |
    | --- | --- | --- |
    | **Logistic Regression** | *0.923077* | *0.96* |
    | **Random Forest** | *0.834783* | *0.96* |
    | **Adaboost** | *0.770492* | *0.94* |



    ''')

    st.title('Try my model!')


    st.write('\n Input a title and selftext below and my model will tell you which subreddit it comes from.')
    pickle_file = open('project3.pkl','rb')
    model = pickle.load(pickle_file)


    title_text = st.text_input("Please enter the title")
    self_text = st.text_input("Please enter the selftext, if there is none, just leave it blank")

    if st.button("Predict"):
        df = input_to_df(title_text,self_text)
        final_df = prepare(df)
        predicted_result = model.predict(final_df['title_selftext_new'])

        st.write(f'The post you entered is from {predicted_result[0]} subreddit')
if page == 'Speech and Pictures Transformation':
    st.markdown('''
    # Noise Detection & Classify

### Description
For this project, using CNN to classify different noise. Not only on the noise itself, but also try to classify them when the nosises combine with my voice.

#### Data Source

[UrbanSound8K:]('https://urbansounddataset.weebly.com/urbansound8k.html')

---

### Data info

- 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes.
    * air_conditioner.
    * car_horn.
    * children_playing.
    * dog_bark.
    * drilling.
    * engine_idling.
    * gun_shot.
    * jackhammer.
    * siren.
    * stree_music


---

### Data EDA/ Preprocessing/ Visualization

- Plot wave plot for each class of sound.
- Extract MFCC features from each sound
- Combined with my voice

### Model building:

Built 2 models with `CNN`:

> First Model, classify the noise only, without my voice

| Data | Accuracy | Loss Value |
| --- | --- | --- |
| **Training Data** | *97.73%* | *0.075* |
| **Testing Data** | *0* | *0.9136* |



### Model results:

> Second Model, classify the noise which is combined with my voice

| Data | Accuracy | Loss Value |
| --- | --- | --- |
| **Training Data** | *90.42%* | *0.298* |
| **Testing Data** | *85.12%* | *0.456* |


### Conclusion

CNN works perfectly with MFCC feature when dealing with sound.
However, there are many need to be improve:
1) The model can only classify one noise at a time, (one reponse variable only), I'll try to make it works on detect&classify multiple noise at a time
2) Try to combine more human's voice, (and different gender since the voice's frequency varies a lot among genders)
3) Final project will be able to remove the noise from the sound track.


Credit to:

1) [Understanding Audio data, Fourier Transform, FFT and Spectrogram features for a Speech Recognition System]('https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520')

2) [Mel Frequency Cepstral Coefficient (MFCC) tutorial]('http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/')

3) [Sound Classification using Deep Learning]('https://mikesmales.medium.com/sound-classification-using-deep-learning-8bc2aa1990b7')
''')
