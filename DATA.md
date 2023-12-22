# Data Description
## Data files
We provide three data files: `twitter_data.json`, `twitter_data.csv`, and `twitter_data_with_gpt.csv`. 
- `twitter_data.csv`: Most people will only need this file, which they can read directly into a Pandas dataframe. 
- `twitter_data_with_gpt.csv`: This file includes the same columns as `twitter_data.csv`, in addition to columns of the form `GPT_*` corresponding to GPT-4 judgments of these keys (e.g. `GPT_angry` is the GPT-4 judgment of whether the tweet expresses anger). Any tweets that GPT-4 did not return valid responses to are filtered out.
- `twitter_data.json`: the original format of our dataset that our code was built on top of. In `utils.py`, we have a function `load_to_df` that converts this JSON into the dataframe corresponding to `twitter_data.csv`. All our Jupyter notebooks start with calling `load_to_df` on `twitter_data.json`. Most people will not need this file and can just use `twitter_data.csv` directly.

## Keys in `twitter_data.csv`
Each row in `twitter_data.csv` corresponds to a tweet for a given user’s timeline. Below we describe what each column in the provided CSV corresponds to.

### timeline
Corresponds to the Timeline type that the tweet was fetched from and can be Engagement, Chronological, Stated Preference, or Stated Preference with Downranking. Engagement and Chronological correspond to the timeline that the tweet was fetched from during the collection of the user’s timeline. 

The Stated Preference timeline, described in our paper, corresponds to an alternative, hypothetical timeline ranked by the user’s stated preference. The Stated Preference with Downranking timeline is a variant of the Stated Preference timeline that also downranks tweets based on the out-group animosity indicated by the user (see Appendix E9). The probability the tweet is included in either of these timelines is included in the column `prob_in_tl`.

### rank
The tweet’s rank in the given user timeline. This should be ignored for the Stated Preference and Stated Preference with Downranking timelines.
### prob_in_tl
The probability of a tweet being in the Stated Preference or Stated Preference with Downranking timelines. This probability is included because ties among tweets with the same score are broken at random.

This is not relevant for the Engagement or Chronological timelines and is set to 1 for tweets from those timelines.

### user_id
Anonymized ID for the CloudResearch Connect user.
### user_summary_leaning
The political leaning of a user, either `LEFT` (mapping to -1) or `RIGHT` (1). We use the user’s self-described political leaning if it takes on Left, Far left, Right, or Far right. If they stated their political leaning was Moderate, we asked whether they lean more towards the Left or Right as of today and used that response.
### collected_at
Timestamp at which the tweet was scraped from the user’s timeline.
### shown_to_user
Boolean value for whether the tweet was shown to the user in the survey. Only the top ten tweets in the engagement/chronological timeline were shown to the user. If a tweet was not shown to a user, then the tweet-level survey responses are NaN.
### user_race
A list of strings corresponding to the user’s race(s). They could select one or multiple from ‘American Indian or Alaska Native’, ‘Asian’, ‘Black or African American’, ‘Native Hawaiian or Pacific Islander’, ‘White’, or ‘Other’.
### user_ethnicity
A list of strings corresponding to the user’s ethnicities. They could select one or multiple from ‘Hispanic’, ‘Latino’, ‘Spanish’, or ‘None of these’.
### user_summary_race
String corresponding to the summary combined race + ethnicity of the user. This key was created for comparison with the ANES 2020 study and uses the same categories: 'Multiple races, non-Hispanic', 'Black or African American', 'Hispanic', 'White', 'Asian or Native Hawaiian/other Pacific Islander' or 'American Indian/Alaska Native or Other'.
### user_gender
String corresponding to the user’s gender. They could select from 'Man', 'Woman', 'Non-binary', or 'Other'.
### user_ideo_political_party
String corresponding to the user’s political party. They could select from 'Independent', 'Democrat', 'Republican', or 'Something else'.
### user_pol_party_further
String corresponding to the political party that the user leans towards, shown to users that selected ‘Independent’ or ‘Something else’ in `user_ideo_political_party`. Limited to selecting from ‘The Democratic Party’ or ‘The Republican Party’.
### user_summary_party
String corresponding to the political party that the user leans towards, limited to selecting from 'Republican' or 'Democrat'.
### user_ideo_political_leaning
String corresponding to the user’s ideological political leaning. They could select from 'Far right', 'Right', 'Moderate', 'Left', 'Far left', or 'Other'.
### user_pol_leaning_further
String corresponding to the user’s political leaning, shown to users that selected ‘Moderate’ or ‘Other’ in `user_ideo_political_leaning`. Limited to selecting from ‘Towards the Left’ or ‘Towards the Right’.
### user_summary_leaning_text
String corresponding to the political ideology that the user leans towards, limited to selecting from 'Right-leaning' or 'Left-leaning'.
### user_education_level
The highest level of schooling that the user has completed, can take on 'Some high school', 'High school graduate', ‘Bachelor's degree’, 'Associate degree', ‘Master's degree or above’, 'Other', or 'Prefer not to answer'.
### user_main_why_tweet
The main reason that the user uses Twitter, can take on 'A way to stay informed', 'Entertainment', 'Keeping me connected to other people', ‘It's useful for my job or school’,      'Lets me see different points of view', or 'A way to express my opinions'.
### user_age_group
The age group of the user, can take on '18-24 years old', '25-34 years old', '35-44 years old', '45-54 years old', '55-64 years old', '65-74 years old', or '75 years or older'.
### user_annual_household_income
The annual household income of the user, can take on 'Less than $25,000', '$25,000 - $50,000', '$50,000 - $100,000', '$100,000 - $200,000', 'More than $200,000', or 'Prefer not to say'.
### content_category
User’s response of the primary content categories that the user’s tweets correspond to, could select up to two from 'Entertainment', 'Hobbies', 'News', 'Other', 'Politics', or 'Work'.
### author_angry
The user’s judgment of how angry the author was feeling in their tweet on a Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### author_anxious
The user’s judgment of how anxious the author was feeling in their tweet on a Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### author_happy
The user’s judgment of how happy the author was feeling in their tweet on a Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### author_sad
The user’s judgment of how sad the author was feeling in their tweet on a Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### reader_angry
The user’s judgment of how angry they felt reading the tweet on a Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### reader_anxious
The user’s judgment of how anxious they felt reading the tweet on a Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### reader_happy
The user’s judgment of how happy they felt reading the tweet on a Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### reader_sad
The user’s judgment of how sad they felt reading the tweet on a Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### is_political
Boolean value corresponding to whether the user thought a tweet was about a political or social issue.
### political_leaning
The user’s judgment of the political leaning of the tweet on a Likert scale of 'Far left' (-2), 'Left' (-1), 'Moderate' (0), 'Right' (1), or 'Far right' (2). This field is represented as the integer value corresponding to the selected Likert score. Can be NaN for non-political tweets.

### partisanship
Integer value in the range [0, 2] corresponding to the absolute value of the tweet’s political leaning (see `political_leaning`).
### ingroup_tweet
Boolean value for whether the tweet belongs to the user’s ingroup.
### outgroup_tweet
Boolean value for whether the tweet belongs to the user’s outgroup.
### political_affect_left
The user’s judgment of how the tweet made them feel about people or groups on the Left on a Likert scale of 'Much worse' (-2), 'Worse' (-1), 'The same as before' (0), 'Better' (1), or 'Much better' (2). This field is represented as the integer value corresponding to the selected Likert score. Can be NaN for non-political tweets.
### political_affect_right
The user’s judgment of how the tweet made them feel about people or groups on the Right on a Likert scale of 'Much worse' (-2), 'Worse' (-1), 'The same as before' (0), 'Better' (1), or 'Much better' (2). This field is represented as the integer value corresponding to the selected Likert score. Can be NaN for non-political tweets.
### ingroup_affect
The user’s judgment of how the tweet made them feel about people or groups in their ingroup on a Likert scale of 'Much worse' (-2), 'Worse' (-1), 'The same as before' (0), 'Better' (1), or 'Much better' (2). This field is represented as the integer value corresponding to the selected Likert score. 
### outgroup_affect
The user’s judgment of how the tweet made them feel about people or groups in their outgroup on a Likert scale of 'Much worse' (-2), 'Worse' (-1), 'The same as before' (0), 'Better' (1), or 'Much better' (2). This field is represented as the integer value corresponding to the selected Likert score. 
### political_outgroup_anger_left
Boolean value calculated from binarizing the user’s response to whether the tweet expresses anger, frustration, or hostility towards a person or group on the Left. This question is only asked if `political_leaning` is 'Right' or 'Far right'.
### political_outgroup_anger_right
Boolean value calculated from binarizing the user’s response to whether the tweet expresses anger, frustration, or hostility towards a person or group on the Right. This question is only asked if `political_leaning` is 'Left' or 'Far left'.
### outgroup_animosity_to_ingroup
Integer value taking on 0 or 1 for whether the tweet expresses anger, frustration, or hostility towards a person or group in the user’s ingroup.
### outgroup_animosity_to_outgroup
Integer value taking on 0 or 1 for whether the tweet expresses anger, frustration, or hostility towards a person or group in the user’s outgroup.
### value
The user’s stated preference for whether they would like to be shown tweets like this one on a scale of 'No' (-1), 'Indifferent' (0), 'Yes' (1). This field is represented as the integer value corresponding to the selected score.
### id_str
ID string of the tweet.
### full_text
String of the full text in the tweet at the time of collection.
### favorite_count
Number of favorites that the tweet had at the time of collection.
### retweet_count
Number of retweets that the tweet had at the time of collection.
### created_at
Timestamp that the tweet was created.
### video_count
Number of videos on the tweet.
### photo_count
Number of photos on the tweet.
### animated_gif_count
Number of animated GIFs on the tweet.
### author_followers_count
Number of followers that the author of the tweet had at the time of collection.
### is_author_verified
Boolean value for whether the author was verified at the time of collection.
### is_following
Boolean value for whether the user was following the author of the tweet.
### display_name
Display name of the tweet author.
### screen_name
Screen name of the tweet author.
### user_id_str
ID string of the tweet author’s Twitter account.
### tweet_type
String indicating the type of the tweet object. Each row corresponds to a main tweet collected from the user timeline or a tweet that is quoted or replied to, respectively corresponding to 'main_tweet', 'quoted_tweet', or 'replied_tweet'.
### main_tweet_id
ID of the main tweet that the quoted or replied tweet is referring to if this row corresponds to a quoted or replied tweet (see `tweet_type`). Otherwise if this row corresponds to a main tweet, then it is the same as the `id_str`.
### external_urls
List of stringified external URLs referenced in the tweet.
### external_url_count
Count of external URLs referenced in the tweet. See `external_urls`.
### full_text_len
Length of the full text of the tweet. See `full_text`.
### tweet_age
Difference between the time that the tweet was created and collected from the user timeline in minutes.
### outgroup_animosity
Boolean value for whether the tweet expresses anger, frustration, or hostility towards a person or group on either the Left or the Right.
### user_time_key
String of the format `{user_id}_{collected_at}` using the corresponding columns.

## Additional keys in `twitter_data_with_gpt.csv`
Below we include the keys for GPT-4 estimates of average treatment effects. This CSV only contains rows with valid GPT responses. Appendix E6 in our paper includes the prompts that we used to query GPT-4.
### GPT_happy
GPT-4 estimates of the `author_happy` key with the same Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### GPT_sad
GPT-4 estimates of the `author_sad` key with the same Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### GPT_angry
GPT-4 estimates of the `author_angry` key with the same Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### GPT_anxious
GPT-4 estimates of the `author_anxious` key with the same Likert scale of 'Not at all', 'Slightly', 'Somewhat', 'Moderately', or 'Extremely.' This field is represented as an integer in the range [0, 4] corresponding to this Likert scale.
### GPT_is_political
GPT-4 estimates of the `is_political` key corresponding to whether the user thought a tweet was about a political or social issue. Could take on 0 ('No') or 1 ('Yes').
### GPT_political_leaning
GPT-4 estimates of the `political_leaning` key corresponding the user’s judgment of the political leaning of the tweet on a Likert scale of 'Far left' (-2), 'Left' (-1), 'Moderate' (0), 'Right' (1), or 'Far right' (2). Could be NaN for tweets that are not about political or social issues.
### GPT_animosity_left
GPT-4 estimates of animosity towards the left, takes on an integer value taking on 0 ('No') or 1 ('Yes') for whether the tweet expresses anger, frustration, or hostility towards the political left. Could be NaN for tweets that are not about political or social issues.
### GPT_animosity_right
GPT-4 estimates of animosity towards the right, takes on an integer value taking on 0 ('No') or 1 ('Yes') for whether the tweet expresses anger, frustration, or hostility towards the political right. Could be NaN for tweets that are not about political or social issues.
