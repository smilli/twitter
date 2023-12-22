import json
from datetime import datetime
from enum import Enum
from typing import Callable, Literal

import numpy as np
import pandas as pd
import tqdm
from cryptorandom.cryptorandom import SHA256
from numpy.random import RandomState
from scipy import stats
from statsmodels.stats import multitest

prng = RandomState(42)

LEFT = -1
RIGHT = 1
EMOTION_KEYS = [
    "author_angry",
    "author_anxious",
    "author_sad",
    "author_happy",
    "reader_angry",
    "reader_anxious",
    "reader_sad",
    "reader_happy",
]
EMOTION_DICT = {"Not at all": 0, "Slightly": 1, "Somewhat": 2, "Moderately": 3, "Extremely": 4}
VALUE_DICT = {"No": -1, "Indifferent": 0, "Yes": 1}


class Timeline(Enum):
    ENGAGEMENT = "Engagement"
    CHRONOLOGICAL = "Chronological"
    STATED_PREFERENCE = "Stated Preference"
    STATED_PREFERENCE_DOWNRANKING = "Stated Preference with Downranking"


def get_user_leaning(user: dict) -> int:
    """
    Returns the political leaning of a user, either LEFT or RIGHT.
    """
    pol_leaning = user["ideo_political_leaning"]
    if pol_leaning == "Left" or pol_leaning == "Far left":
        return LEFT
    elif pol_leaning == "Right" or pol_leaning == "Far right":
        return RIGHT
    elif pol_leaning == "Moderate" or pol_leaning == "Other":
        pol_leaning_further = user["ideo_political_leaning_further"]
        if pol_leaning_further == "Towards the Left":
            return LEFT
        elif pol_leaning_further == "Towards the Right":
            return RIGHT
        else:
            raise ValueError(f"Invalid political further leaning: {pol_leaning_further}")
    else:
        raise ValueError(f"Invalid political leaning: {pol_leaning}")


def get_user_party(user: dict) -> int:
    """
    Returns the political party of a user, either Republican or Democrat.
    """
    pol_party = user["ideo_political_party"]
    if pol_party not in ["Democrat", "Republican"]:
        pol_party_further = user["ideo_political_party_further"]
        if pol_party_further == "The Democratic Party":
            return "Democrat"
        elif pol_party_further == "The Republican Party":
            return "Republican"
        else:
            raise ValueError(f"Invalid political party further: {pol_party_further}")
    else:
        return pol_party
    
def get_tweet_values(tweet_objs: list[dict]) -> list:
    """
    Returns a list of values for each tweet object in tweet_objs.

    Params:
        tweet_objs: list of tweet objects loaded from json

    Returns:
        list of values for each tweet object, values can be either
        1, 0.5, 0, -0.5, -1
    """
    values = []  # user stated value for each tweet
    for tweet_obj in tweet_objs:
        main_tweet = tweet_obj["main_tweet"]
        main_tweet_value = VALUE_DICT[main_tweet["value"]]
        value = main_tweet_value
        # if there is a replied or quoted tweet, the the value
        # of the overall tweet object is the average of the value
        # of the replied / reply tweet or quoted / quote tweet
        for key in ["replied_tweet", "quoted_tweet"]:
            if key in tweet_obj:
                other_tweet = tweet_obj[key]
                other_tweet_value = VALUE_DICT[other_tweet["value"]]
                value = (main_tweet_value + other_tweet_value) / 2
        values.append(value)
    values = np.array(values)
    return values


def get_animosity_indices(tweet_objs: list[dict]) -> list:
    """
    Returns a list of indices of tweets that have out-group animosity.

    Params:
        tweet_objs: list of tweet objects loaded from json

    Returns:
        list of indices of tweets that have out-group animosity
    """
    animosity_indices = []
    for i, tweet_obj in enumerate(tweet_objs):
        for key in ["main_tweet", "replied_tweet", "quoted_tweet"]:
            if key in tweet_obj:
                other_tweet = tweet_obj[key]
                if other_tweet["political_outgroup_anger_left"] or other_tweet["political_outgroup_anger_right"]:
                    animosity_indices.append(i)
                    break
    return animosity_indices


def add_stated_pref_labels(data: list, downrank_outgroup_animosity=False, num_tweets_in_tl=10) -> list:
    """
    Add labels to data loaded from json file indicating the prob that
    the tweet would be in a timeline ranked by the user's stated preference.

    Params:
        data: list of dicts loaded from json file
        downrank_outgroup_animosity: whether to downrank tweets with out-group animosity
        num_tweets_in_tl: number of tweets in the stated preference timeline

    Returns:
        list of dicts with added labels
    """
    for user in data:
        tweet_objs = get_unique_tweet_objects(user)
        values = get_tweet_values(tweet_objs)
        if downrank_outgroup_animosity:
            tweet_inds_with_animosity = get_animosity_indices(tweet_objs)
            if len(tweet_inds_with_animosity) > 0:
                values[tweet_inds_with_animosity] = values[tweet_inds_with_animosity] - 0.5

        pos_values = [1.0, 0.5, 0.0, -0.5, -1.0, -1.5]
        counts = [np.sum(values == v) for v in pos_values]

        # calculate probs for each tweet to be in the stated preference timeline
        probabilities = np.zeros(len(values))
        for i, (count, value) in enumerate(zip(counts, pos_values)):
            count_sum = sum(counts[: i + 1])
            if count_sum >= num_tweets_in_tl:
                for j in range(i):
                    probabilities[values == pos_values[j]] = 1
                probabilities[values == value] = (num_tweets_in_tl - count_sum + count) / count
                break

        # add labels to tweets
        label = "prob_in_stated_tl" if not downrank_outgroup_animosity else "prob_in_stated_tl_with_downranking"
        for tweet_obj, prob in zip(tweet_objs, probabilities):
            tweet_obj[label] = prob
    return data


def get_unique_tweet_objects(user: dict) -> list:
    """
    Returns a list of unique tweet objects shown to user.

    Params:
        user: dict loaded from json file

    Returns:
        list of unique tweet objects
    """
    tweet_objs = user["tweets"]["personalized"] + user["tweets"]["chronological"]
    unique_tweet_ids = []
    unique_tweet_objs = []
    for tweet_obj in tweet_objs:
        if tweet_obj["shown_to_user"]:
            main_tweet_id = tweet_obj["main_tweet"]["id_str"]
            if main_tweet_id not in unique_tweet_ids:
                unique_tweet_ids.append(main_tweet_id)
                unique_tweet_objs.append(tweet_obj)
    return unique_tweet_objs


def get_anes_races(df: pd.DataFrame) -> list:
    """
    Coerces our race data into the ANES categories. Source: https://electionstudies.org/wp-content/uploads/2022/02/anes_timeseries_2020_userguidecodebook_20220210.pdf
    """
    df = df.copy()  # Prevent any changes from changing the original data

    # Manually iterate over all of the cases.
    df["user_ethnicity"] = df["user_ethnicity"].astype(str)
    df["user_race"] = df["user_race"].astype(str)

    changed = np.zeros(len(df["user_race"]))

    hispanic_mask = df["user_ethnicity"].str.contains("Hispanic|Spanish|Latino")
    df.loc[hispanic_mask, "user_race"] = "Hispanic"

    amerind_alaska_mask = df["user_race"] == "['American Indian or Alaska Native']"
    df.loc[amerind_alaska_mask, "user_race"] = "American Indian/Alaska Native or Other"

    other_mask = df["user_race"] == "['Other']"
    df.loc[other_mask, "user_race"] = "American Indian/Alaska Native or Other"

    # Note that "Asian or Native Hawaiian/other Pacific Islander" in ANES categories is either Asian OR Native Hawaiian/other Pacific Islander but not both.
    # AAPI as in both Asian and Native Hawaiian/other Pacific Islander ends up in multiple races, non-Hispanic according to their codebook.
    native_hawaii_pi_mask = df["user_race"] == "['Native Hawaiian or Pacific Islander']"
    df.loc[native_hawaii_pi_mask, "user_race"] = "Asian or Native Hawaiian/other Pacific Islander"

    asian_mask = df["user_race"] == "['Asian']"
    df.loc[asian_mask, "user_race"] = "Asian or Native Hawaiian/other Pacific Islander"

    black_mask = df["user_race"] == "['Black or African American']"
    df.loc[black_mask, "user_race"] = "Black or African American"

    white_mask = df["user_race"] == "['White']"
    df.loc[white_mask, "user_race"] = "White"

    changed = (
        changed
        + hispanic_mask
        + amerind_alaska_mask
        + other_mask
        + native_hawaii_pi_mask
        + asian_mask
        + black_mask
        + white_mask
    ).astype(bool)
    # Opposite mask of changed
    df.loc[~changed, "user_race"] = "Multiple races, non-Hispanic"

    return df["user_race"]


def load_to_df(filepath: str = "twitter_data.json") -> pd.DataFrame:
    """
    Loads data from json file into a pandas DataFrame.

    Params:
        filepath : str, optional
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    data = add_stated_pref_labels(data, downrank_outgroup_animosity=False)
    data = add_stated_pref_labels(data, downrank_outgroup_animosity=True)
    rows = []

    for user in tqdm.tqdm(data):
        user_summary_leaning = get_user_leaning(user)
        timelines = ["personalized", "chronological"]
        for timeline in timelines:
            tweet_objs = user["tweets"][timeline]
            for tweet_obj in tweet_objs:
                tweet_keys = ["main_tweet", "replied_tweet", "quoted_tweet"]
                prob_in_stated_tl = tweet_obj["prob_in_stated_tl"] if "prob_in_stated_tl" in tweet_obj else 0
                prob_in_stated_tl_with_downranking = (
                    tweet_obj["prob_in_stated_tl_with_downranking"]
                    if "prob_in_stated_tl_with_downranking" in tweet_obj
                    else 0
                )
                collected_at = tweet_obj["tl_collection_time"]
                rank = tweet_obj["rank"]
                for key in tweet_keys:
                    if key in tweet_obj:
                        tweet = tweet_obj[key]
                        tweet["tweet_type"] = key
                        tweet["main_tweet_id"] = tweet_obj["main_tweet"]["id_str"]
                        tweet["external_urls"] = [
                            url_d["expanded_url"]
                            for url_d in tweet["urls"]
                            if "https://twitter.com" not in url_d["expanded_url"]
                        ]
                        tweet["external_url_count"] = len(tweet["external_urls"])
                        del tweet["urls"]
                        del tweet["link_count"]

                        row = {
                            "user_id": user["user_id"],
                            "user_summary_leaning": user_summary_leaning,
                            "collected_at": collected_at,
                            "shown_to_user": tweet_obj["shown_to_user"],
                            "user_race": user["race"],
                            "user_ethnicity": user["spanish_hispanic_latino"],
                            "user_gender": user["gender"],
                            "user_ideo_political_party": user["ideo_political_party"],
                            "user_pol_party_further": user["ideo_political_party_further"],
                            "user_summary_party": get_user_party(user),
                            "user_ideo_political_leaning": user["ideo_political_leaning"],
                            "user_pol_leaning_further": user["ideo_political_leaning_further"],
                            "user_summary_leaning_text": "Left-leaning" if user_summary_leaning == -1 else "Right-leaning",
                            "user_education_level": user["education"],
                            "user_main_why_tweet": user["main_why_tweet"],
                            "user_age_group": user["age_group"],
                            "user_annual_household_income": user["annual_household_income"],
                            "content_category": user["content_category"],
                            **tweet,
                        }

                        if "political_leaning" in tweet:
                            tweet_pol_leaning = tweet["political_leaning"]
                            ingroup_tweet = (user_summary_leaning * tweet_pol_leaning > 0) if tweet_pol_leaning else False
                            outgroup_tweet = (user_summary_leaning * tweet_pol_leaning < 0) if tweet_pol_leaning else False
                            row = {
                                **row,
                                "ingroup_tweet": ingroup_tweet,
                                "outgroup_tweet": outgroup_tweet,
                            }
                        if timeline == "personalized":
                            rows.append({"timeline": "Engagement", "rank": rank, "prob_in_tl": 1, **row})
                        if timeline == "chronological":
                            rows.append({"timeline": "Chronological", "rank": rank, "prob_in_tl": 1, **row})
                        if prob_in_stated_tl > 0:
                            rows.append({"timeline": "Stated Preference", "prob_in_tl": prob_in_stated_tl, **row})
                        if prob_in_stated_tl_with_downranking > 0:
                            rows.append(
                                {
                                    "timeline": "Stated Preference with Downranking",
                                    "prob_in_tl": prob_in_stated_tl,
                                    **row,
                                }
                            )
    df = pd.DataFrame(rows)
    df["timeline"] = df["timeline"].astype("category")
    df["full_text_len"] = df["full_text"].apply(len)
    df["collected_at"] = pd.to_datetime(df["collected_at"], format='mixed')
    df["created_at"] = pd.to_datetime(df["created_at"], format='mixed')
    df["tweet_age"] = (df["collected_at"] - df["created_at"]).dt.total_seconds() / 60
    df["value"] = df["value"].map(VALUE_DICT)
    for key in EMOTION_KEYS:
        df[key] = df[key].map(EMOTION_DICT)
    df["partisanship"] = df["political_leaning"].abs().fillna(0)
    df["political_outgroup_anger_left"] = df["political_outgroup_anger_left"].fillna(0).astype("bool")
    df["political_outgroup_anger_right"] = df["political_outgroup_anger_right"].fillna(0).astype("bool")
    df["outgroup_animosity"] = (df["political_outgroup_anger_left"] | df["political_outgroup_anger_right"]).astype(
        "int"
    )
    df["outgroup_animosity_to_ingroup"] = np.where(
        df["user_summary_leaning"] == LEFT, df["political_outgroup_anger_left"], df["political_outgroup_anger_right"]
    )
    df["outgroup_animosity_to_outgroup"] = np.where(
        df["user_summary_leaning"] == LEFT, df["political_outgroup_anger_right"], df["political_outgroup_anger_left"]
    )
    df["outgroup_animosity_to_ingroup"] = df["outgroup_animosity_to_ingroup"].fillna(0).astype("int")
    df["outgroup_animosity_to_outgroup"] = df["outgroup_animosity_to_outgroup"].fillna(0).astype("int")
    df["ingroup_affect"] = np.where(
        df["user_summary_leaning"] == LEFT, df["political_affect_left"], df["political_affect_right"]
    )
    df["ingroup_affect"] = df["ingroup_affect"].fillna(0)
    df["outgroup_affect"] = np.where(
        df["user_summary_leaning"] == LEFT, df["political_affect_right"], df["political_affect_left"]
    )
    df["outgroup_affect"] = df["outgroup_affect"].fillna(0)
    df["user_time_key"] = df["user_id"].astype(str) + "_" + df["collected_at"].astype(str)

    df["user_summary_race"] = get_anes_races(df)

    return df


def get_bootstrap_ci(pers_means: np.array, chron_means: np.array, std: float, n: int = 10000) -> tuple:
    """
    Returns the bootstrap confidence interval for the difference in means
    between the engagement and chronological timelines.

    Params:
        pers_means: The mean outcomes in the personalized timelines.
        chron_means: The mean outcomes in the chronological timelines.
        std: The standard deviation to use for normalizing the effect.
        n: The number of bootstrap samples to take.

    Returns:
        A tuple of the lower and upper bounds of the bootstrap confidence interval.
    """

    def statistic(eng, chron, axis):
        diff = np.mean(eng - chron, axis=axis)
        stats = diff / std
        return stats

    ci = stats.bootstrap(
        data=(pers_means, chron_means),
        statistic=statistic,
        n_resamples=n,
        method="basic",
        paired=True,
        random_state=prng,
    ).confidence_interval

    return ci.low, ci.high


def get_agg_func(df: pd.DataFrame, data_col: str, weight_col: str) -> Callable:
    """
    Returns a function that calculates the weighted average of the given `data_col`
    in `df` using the weights in `weight_col`.
    """

    def weighted_average(x: pd.Series):
        """Calculate the weighted average of the specific column."""
        probs = df.iloc[x.index][weight_col]
        return np.average(x, weights=probs)

    return weighted_average


def get_mean_by_user_time_key_and_tl(df: pd.DataFrame, keys: list[str]):
    """
    Returns a new dataframe with the given keys grouped by user_time_key and timeline
    and aggregated by the weighted average of the probability of the tweet being in the timeline.
    """
    df = df.reset_index(drop=True)

    aggregations = {}
    for col in keys:
        aggregations[col] = (col, get_agg_func(df, col, "prob_in_tl"))

    # Add in user_id
    keys = keys + ["user_id"]
    aggregations["user_id"] = ("user_id", lambda x: x.iloc[0])

    grouped = df.groupby(["user_time_key", "timeline"])[keys].agg(**aggregations)
    return grouped


def get_mean_by_user_id_and_tl(df: pd.DataFrame, keys: list[str], grouped=None):
    """
    Returns a new dataframe with the given keys grouped by user_id and aggregated
    by the weighted average of the probability of the tweet being in the timeline.
    Grouped is an optional argument to avoid recomputation if the user_time_key_tl aggregation is already done
    """
    grouped = get_mean_by_user_time_key_and_tl(df, keys) if grouped is None else grouped
    return grouped.groupby(["user_id", "timeline"]).mean()


def get_mean_by_tl(df: pd.DataFrame, keys: list[str]):
    """
    Returns a new dataframe with the given keys grouped by user_id and timeline
    and aggregated using the weighted average of the probability of the tweet being in the timeline.
    """
    grouped = get_mean_by_user_id_and_tl(df, keys)
    return grouped.groupby("timeline").mean()

def get_resp_dist_for_user_id_and_tl(df: pd.DataFrame, key: str, scale: list[int]):
    """
    Returns a distribution of survey responses for the given keys for each user_id and timeline pair.
    Groups by user_id and timeline type for each `effect` and returns an average over (user_id, tl) pairs.
    
    Params:
        df: dataframe for a specific timeline type
        key: effect column to return response distribution for
        scale: range of values that the specified `key` can take on    
    Returns:
        DataFrame with response distribution for the specified `key` post
        grouping over (user id, timeline type) pair and averaging over these pairs
    """       
    # group by (user_id, timeline) and key, then count the number of occurrences
    grouped = df.groupby(['user_id', 'timeline', key]).size().reset_index(name='count')

    # change effect vals to columns
    pivoted = grouped.pivot_table(index=['user_id', 'timeline'], columns=key, values='count', fill_value=0)

    # calculate sum along rows to get the total count per (user_id, timeline) pair
    pivoted['total'] = pivoted.sum(axis=1)

    # calculate % dist for each likert score in the provided scale
    for col in scale:
        pivoted[col] = (pivoted[col] / pivoted['total']) * 100

    # drop 'total' column
    pivoted.drop('total', axis=1, inplace=True)

    result_df = pivoted.reset_index()
    return result_df

def get_resp_dist_for_tl(df: pd.DataFrame, key: str, tl_type: Timeline, scale: list[int]):
    """
    Returns response distribution for `key` and timeline type after grouping
    by user_id and timeline type for each effect `key` and averaging over (user_id, tl) 
    pairs.
    
    Params:
        df: dataframe containing data of user survey responses to all tweets
        key: effect column to return response distribution for
        tl_type: timeline type to aggregate over
        scale: range of values that the specified `key` can take on    
    Returns:
        DataFrame with response distribution for the specified `key` and timeline type post
        grouping over (user id, timeline type) pair and averaging over these pairs
    """
    assert tl_type.value in (Timeline.CHRONOLOGICAL.value, Timeline.ENGAGEMENT.value), \
        f"Timeline type must be Chronological or Engagement but is {tl_type.value}"

    dist_df = get_resp_dist_for_user_id_and_tl(df, key, scale)
    dist_df = dist_df[dist_df['timeline'] == tl_type.value].mean(numeric_only=True).to_frame()

    # reset index and rename columns
    dist_df = dist_df.reset_index()
    dist_df = dist_df.rename(columns={key: 'likert_score', 0: 'percent'})

    # format likert score dist
    dist_df['likert_score'] = dist_df['likert_score'].astype(int)

    return dist_df


def get_empty_effect_df(key_name) -> pd.DataFrame:
    return {
        "Outcome": key_name,
        "Standardized Effect": np.NaN,
        "Unstandardized Effect": np.NaN,
        "Chron Mean": np.NaN,
        "Pers Mean": np.NaN,
        "p-value": np.NaN,
        "Confidence Interval (standardized)": np.NaN,
        "Confidence Interval (unstandardized)": np.NaN,
    }


def get_prng(seed=None):
    """
    NOTE: This is from the permute.core library
    Turn seed into a cryptorandom instance

    Parameters
    ----------
    seed : {None, int, str, RandomState}
        If seed is None, return generate a pseudo-random 63-bit seed using np.random
        and return a new SHA256 instance seeded with it.
        If seed is a number or str, return a new cryptorandom instance seeded with seed.
        If seed is already a numpy.random RandomState or SHA256 instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    RandomState
    """
    if seed is None:
        # Need to specify dtype (Windows defaults to int32)
        seed = np.random.randint(0, 10**10, dtype=np.int64)  # generate an integer
    if seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer, float, str)):
        return SHA256(seed)
    if isinstance(seed, (np.random.RandomState, SHA256)):
        return seed
    raise ValueError("%r cannot be used to seed cryptorandom" % seed)


def one_sample(x, reps=10**5, stat="mean", alternative="greater", keep_dist=False, seed=None, plus1=True):
    r"""
    NOTE: This is a modified implementation of one_sample from the permute.core library
    One-sided or two-sided, one-sample permutation test for the mean,
    with p-value estimated by simulated random sampling with
    reps replications.

    Tests the hypothesis that x is distributed symmetrically symmetric about 0
    against the alternative that x comes from
    a population with mean

    (a) greater than 0,
        if side = 'greater'
    (b) less than 0,
        if side = 'less'
    (c) different from 0,
        if side = 'two-sided'

    If ``keep_dist``, return the distribution of values of the test statistic;
    otherwise, return only the number of permutations for which the value of
    the test statistic and p-value.

    Parameters
    ----------
    x : array-like
        Sample 1, pd.DataFrame with two columns, 'id', which corresponds to the user_id of the tweet,
        and 'diff', which is the difference in means between the engagement and chronological timelines.
    reps : int
        number of repetitions
    stat : {'mean', 't'}
        The test statistic. The statistic is computed based on z = x

        (a) If stat == 'mean', the test statistic is mean(z).
        (b) If stat == 't', the test statistic is the t-statistic--
            but the p-value is still estimated by the randomization,
            approximating the permutation distribution.
        (c) If stat is a function (a callable object), the test statistic is
            that function.  The function should take a permutation of the
            data and compute the test function from it. For instance, if the
            test statistic is the maximum absolute value, $\max_i |z_i|$,
            the test statistic could be written:

            f = lambda u: np.max(abs(u))
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test
    keep_dist : bool
        flag for whether to store and return the array of values
        of the irr test statistic
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator
    plus1 : bool
        flag for whether to add 1 to the numerator and denominator of the
        p-value based on the empirical permutation distribution.
        Default is True.

    Returns
    -------
    float
        the estimated p-value
    float
        the test statistic
    list
        The distribution of test statistics.
        These values are only returned if `keep_dist` == True
    """
    prng = get_prng(seed)

    z = x

    thePvalue = {
        "greater": lambda pUp, pDn: pUp + plus1 / (reps + plus1),
        "less": lambda pUp, pDn: pDn + plus1 / (reps + plus1),
        "two-sided": lambda pUp, pDn: 2 * np.min([0.5, pUp + plus1 / (reps + plus1), pDn + plus1 / (reps + plus1)]),
    }
    stats = {"mean": lambda u: np.mean(u)}
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    tst = tst_fun(z)
    n = len(z)
    true_diff = z["diff"]

    if keep_dist:
        dist = []
        for i in range(reps):
            # hardcoded to our assumed use case
            z["diff"] = true_diff * (1 - 2 * prng.randint(0, 2, n))
            dist.append(tst_fun(z))
        pUp = np.sum(dist >= tst) / (reps + plus1)
        pDn = np.sum(dist <= tst) / (reps + plus1)
        return thePvalue[alternative](pUp, pDn), tst, dist
    else:
        hitsUp = 0
        hitsDn = 0
        for i in range(reps):
            # hardcoded to our assumed use case
            z["diff"] = true_diff * (1 - 2 * prng.randint(0, 2, n))
            tv = tst_fun(z)
            hitsUp += tv >= tst
            hitsDn += tv <= tst
        pUp = hitsUp / (reps + plus1)
        pDn = hitsDn / (reps + plus1)
        return thePvalue[alternative](pUp, pDn), tst


def get_effects_keys(
    df: pd.DataFrame,
    timeline: Timeline,
    keys: list,
    key_names: list = None,
    stds: list = None,
) -> pd.DataFrame:
    """
    Returns a new dataframe with results on algorithmic amplification on the given timeline
    relative to the chronological baseline
    for each key in keys. The standard deviation to use for normalizing the effect can be provided in `stds`.

    Params:
        df: The dataframe to use.
        timeline: Timeline type.
        keys: The keys in df to compute effects for.
        key_names: The names to give the keys for the new returned df.
        stds: The standard deviations to use for normalizing the effects.

    Returns:
        A new dataframe of unstandardized, standardized effects, CIs, and p-values.
    """    
    def grouped_mean(means: pd.DataFrame):
        """
        Takes in a pd.DataFrame with columns "id" and "diff", user_ids and difference in means respectively,
        and calculates the mean of each user and then the mean of all of the user means.
        """
        return means.groupby("id").mean().mean()["diff"]
    
    if len(df) == 0:
        data = [get_empty_effect_df(key_name) for key_name in key_names]
        return pd.DataFrame(data)
    
    # aggregate columns through weighted average where weights are the probability that the tweet was in the timeline
    # for chron and engagement, the weights are always 1, but for stated pref they may be in (0, 1].
    grouped = get_mean_by_user_time_key_and_tl(df, keys)
    pivot_df = grouped.reset_index().pivot(index="user_time_key", columns="timeline", values=keys)
    ids = pivot_df.index.str.split("_").str[0]
    bootstrap_df = get_mean_by_user_id_and_tl(df, keys, grouped=grouped)

    if stds is None:
        # We use the standard deviation of the chronological timeline instead of the pooled stdv (as in Cohen's d)
        # because we need to calculate the effect of multiple timelines (engagement, stated preference).
        # Using the standard deviation of the chronological timeline (the control) allows comparison between
        # effects for both engagement + stated preferences timeline. This is also called Glass's Delta and was argued
        # for by Glass (1976) for the same reason.
        stds = df[df["timeline"] == "Chronological"].groupby(["user_id"])[keys].mean().std().tolist()

    if key_names is None:
        key_names = keys

    results = []
    for key, key_name, std in zip(keys, key_names, stds):
        pers_means = pivot_df[key][timeline.value]
        chron_means = pivot_df[key]["Chronological"]

        # Diffs for permutation test
        diff_vals = pers_means - chron_means
        diffs = pd.DataFrame({"id": ids, "diff": diff_vals})

        # User aggregated means for bootstrap ci
        pers_means_user = bootstrap_df[key].xs(timeline.value, level="timeline")
        chron_means_user = bootstrap_df[key].xs("Chronological", level="timeline")

        if len(pers_means_user) <= 1 or len(chron_means_user) <= 1 or std == 0:
            # Not enough data to estimate a boostrap CI
            results.append(get_empty_effect_df(key_name))
            continue

        # permutation test
        p, tst = one_sample(diffs, stat=grouped_mean, alternative="two-sided", reps=10000, seed=prng)

        # calculate bootstrap confidence interval
        std_ci = get_bootstrap_ci(pers_means_user, chron_means_user, std)
        unstd_ci = get_bootstrap_ci(pers_means_user, chron_means_user, 1)
        curr_results = {
            "Outcome": key_name,
            "Standardized Effect": tst / std,
            "Unstandardized Effect": tst,
            "Chron Mean": chron_means.mean(),
            "Pers Mean": pers_means.mean(),
            "p-value": p,
            "Confidence Interval (standardized)": std_ci,
            "Confidence Interval (unstandardized)": unstd_ci,
        }
        results.append(curr_results)
    return pd.DataFrame(results)


def get_effects(df: pd.DataFrame, timeline: Timeline, N: int = -1) -> pd.DataFrame:
    """
    Returns a new dataframe with effects of the given timeline
    (Timeline.ENGAGEMENT or Timeline.STATED_PREFERENCE) relative to the chronological baseline
    on the pre-registered outcomes (Figure 1 in paper).

    Params:
        df: The dataframe of data (is loaded from json using the `load_to_df` func).
        timeline: The timeline to calculate effects for.
        N: Number of tweets to calculate the effects for. If N = -1, then use all of the tweets in the timeline.

    Returns:
        A new dataframe of unstandardized, standardized effects, CIs,
        and p-values for all pre-registered outcomes.
    """
    ## Get effects for non-political outcomes (emotions + value) for tweets overall ##
    non_pol_keys = EMOTION_KEYS + ["value"]
    non_pol_key_names = [
        "Author Angry",
        "Author Anxious",
        "Author Sad",
        "Author Happy",
        "Reader Angry",
        "Reader Anxious",
        "Reader Sad",
        "Reader Happy",
        "Reader Pref",
    ]

    key_names = [key + " (overall)" for key in non_pol_key_names]
    if N != -1:
        df = df[df["rank"] <= N]

    effects_df = get_effects_keys(df, timeline, non_pol_keys, key_names)

    ## Get effects for non-political outcomes (emotions + value) for just political tweets ##
    # Get stds of all keys for *all* tweets
    stds = df[df["timeline"] == "Chronological"].groupby(["user_id"])[non_pol_keys].mean().std().tolist()
    # Create df that restricts to political tweets only
    pol_df = df[df["is_political"] == True]
    # Filter to user_time_keys that have at least one political tweet in both the eng and chron timelines
    group_counts = pol_df.groupby(["user_time_key", "timeline"]).size().unstack(fill_value=0)
    mask = (group_counts[timeline.value] > 0) & (group_counts["Chronological"] > 0)
    filtered_user_time_keys = group_counts[mask].index
    filtered_pol_df = pol_df[pol_df["user_time_key"].isin(filtered_user_time_keys)].reset_index(drop=True)
    key_names = [key + " (political)" for key in non_pol_key_names]
    effects_df = pd.concat([effects_df, get_effects_keys(filtered_pol_df, timeline, non_pol_keys, key_names, stds)])

    ## Get effects for political outcomes ##
    pol_keys = ["partisanship", "outgroup_animosity", "ingroup_affect", "outgroup_affect"]
    key_names = ["Partisanship", "Out-group Animosity", "In-group Perception", "Out-group Perception"]
    effects_df = pd.concat([effects_df, get_effects_keys(df, timeline, pol_keys, key_names)])

    ## Get in-group and out-group perception broken down by user leaning ##
    # Filter df by user political leaning
    left_df = df[df["user_summary_leaning"] == LEFT].reset_index(drop=True)
    right_df = df[df["user_summary_leaning"] == RIGHT].reset_index(drop=True)
    perc_keys = ["ingroup_affect", "outgroup_affect"]
    key_names = ["In-group Perception (left users)", "Out-group Perception (left users)"]
    effects_df = pd.concat([effects_df, get_effects_keys(left_df, timeline, perc_keys, key_names)])
    key_names = ["In-group Perception (right users)", "Out-group Perception (right users)"]
    effects_df = pd.concat([effects_df, get_effects_keys(right_df, timeline, perc_keys, key_names)])

    # Order to make easier for graphing
    outcomes_ordered = (
        list(reversed(["Reader Pref (overall)", "Reader Pref (political)"]))
        + list(
            reversed(
                [
                    "Partisanship",
                    "Out-group Animosity",
                    "In-group Perception",
                    "Out-group Perception",
                    "In-group Perception (left users)",
                    "Out-group Perception (left users)",
                    "In-group Perception (right users)",
                    "Out-group Perception (right users)",
                ]
            )
        )
        + list(
            reversed(
                [
                    "Reader Angry (political)",
                    "Reader Sad (political)",
                    "Reader Anxious (political)",
                    "Reader Happy (political)",
                ]
            )
        )
        + list(
            reversed(
                [
                    "Author Angry (political)",
                    "Author Sad (political)",
                    "Author Anxious (political)",
                    "Author Happy (political)",
                ]
            )
        )
        + list(
            reversed(
                ["Reader Angry (overall)", "Reader Sad (overall)", "Reader Anxious (overall)", "Reader Happy (overall)"]
            )
        )
        + list(
            reversed(
                ["Author Angry (overall)", "Author Sad (overall)", "Author Anxious (overall)", "Author Happy (overall)"]
            )
        )
    )
    effects_df["Outcome"] = pd.Categorical(effects_df["Outcome"], categories=outcomes_ordered, ordered=True)
    effects_df.sort_values("Outcome", inplace=True)
    return effects_df

def get_empty_effect_df(key_name) -> pd.DataFrame:
    return {
        "Outcome": key_name,
        "Standardized Effect": np.NaN,
        "Unstandardized Effect": np.NaN,
        "Chron Mean": np.NaN,
        "Pers Mean": np.NaN,
        "p-value": np.NaN,
        "Confidence Interval (standardized)": np.NaN,
        "Confidence Interval (unstandardized)": np.NaN,
    }


def BKY(p_values: list, alpha: float):
    """
    Benjamimi-Krieger-Yekutieli procedure for controlling false discovery rate at alpha.

    Params:
        p_values: list of p-values
        alpha: desired false discovery rate

    Returns:
        list of rejected hypotheses
    """
    alpha_prime = alpha / (1 + alpha)
    rejected, pvalue_corrected = multitest.fdrcorrection(p_values, alpha=alpha_prime, method="indep", is_sorted=False)
    c = sum(rejected)
    if c != 0:
        M = len(p_values)
        m0 = M - c
        alpha_asterisk = alpha_prime * M / m0
        rejected, pvalue_corrected = multitest.fdrcorrection(
            p_values, alpha=alpha_asterisk, method="indep", is_sorted=False
        )
    return rejected, pvalue_corrected


def get_sharpened_p(p_values: list):
    """
    Sharpened p-values for controlling false discovery rate.

    Params:
        p_values: list of p-values

    Returns:
        list of sharpened p-values
    """
    res_df = pd.DataFrame()
    n = len(p_values)
    for alpha in np.arange(0, 1, 0.0001):
        rejected, _ = BKY(p_values, alpha)
        res_df[alpha] = rejected
    threshold_index = [
        np.where(res_df.iloc[i, :])[0][0] if len(np.where(res_df.iloc[i, :])[0]) != 0 else None for i in range(n)
    ]
    lowest_alpha = [t * 0.0001 if t else None for t in threshold_index]
    return lowest_alpha