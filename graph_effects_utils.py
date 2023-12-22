import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_y_labels() -> list[str]:
    """
    Returns a list of y-axis labels.
    """
    return (
        list(reversed(["Reader Pref (overall)", "Reader Pref (political)"]))
        + list(
            reversed(
                [
                    "Partisanship",
                    "Out-group Animosity",
                    "In-group Perception",
                    "Out-group Perception",
                    "In-group Perc. (left users)",
                    "Out-group Perc. (left users)",
                    "In-group Perc. (right users)",
                    "Out-group Perc. (right users)",
                ]
            )
        )
        + list(
            reversed(["Reader Angry", "Reader Sad", "Reader Anxious", "Reader Happy"])
        )
        + list(
            reversed(["Author Angry", "Author Sad", "Author Anxious", "Author Happy"])
        )
        + list(
            reversed(["Reader Angry", "Reader Sad", "Reader Anxious", "Reader Happy"])
        )
        + list(
            reversed(["Author Angry", "Author Sad", "Author Anxious", "Author Happy"])
        )
    )


def get_y_positions() -> list[int]:
    """
    Gets the positions for where each ylabel should be placed,
    in particular, the indices create four groups of effects:
    emotions (all tweets), emotions (political tweets), political effects,
    and reader stated preference.
    """
    GROUP_OFFSET = 4
    NUM_STATED_PREF_KEYS = 2
    pref_positions = [i for i in range(NUM_STATED_PREF_KEYS)]
    NUM_POLITICAL_KEYS = 8
    political_positions = [
        i
        for i in range(
            GROUP_OFFSET + NUM_STATED_PREF_KEYS,
            GROUP_OFFSET + NUM_STATED_PREF_KEYS + NUM_POLITICAL_KEYS,
        )
    ]
    last_pos = GROUP_OFFSET + NUM_STATED_PREF_KEYS + NUM_POLITICAL_KEYS
    positions = pref_positions + political_positions
    NUM_EMOTIONS = 4
    AUTHOR_READER_OFFSET = 1
    author_positions_overall = [
        i
        for i in range(GROUP_OFFSET + last_pos, GROUP_OFFSET + NUM_EMOTIONS + last_pos)
    ]
    last_pos = GROUP_OFFSET + NUM_EMOTIONS + last_pos
    reader_positions_overall = [
        i
        for i in range(
            AUTHOR_READER_OFFSET + last_pos,
            AUTHOR_READER_OFFSET + NUM_EMOTIONS + last_pos,
        )
    ]
    last_pos = AUTHOR_READER_OFFSET + NUM_EMOTIONS + last_pos
    author_positions_political = [
        i
        for i in range(GROUP_OFFSET + last_pos, GROUP_OFFSET + NUM_EMOTIONS + last_pos)
    ]
    last_pos = GROUP_OFFSET + NUM_EMOTIONS + last_pos
    reader_positions_political = [
        i
        for i in range(
            AUTHOR_READER_OFFSET + last_pos,
            AUTHOR_READER_OFFSET + NUM_EMOTIONS + last_pos,
        )
    ]
    positions = (
        pref_positions
        + political_positions
        + author_positions_overall
        + reader_positions_overall
        + author_positions_political
        + reader_positions_political
    )
    return positions


def reorder_effects_df(effects_df: pd.DataFrame):
    """
    Reorders the effects dataframe so that the effects are grouped into four categories:
    emotions (all tweets), emotions (political tweets), political effects,
    and reader stated preference.

    Params:
        effects_df: A dataframe containing the effects for a given timeline.

    Returns:
        A reordered dataframe.
    """
    outcome_orders = [
        "Author Angry (overall)",
        "Author Sad (overall)",
        "Author Anxious (overall)",
        "Author Happy (overall)",
        "Reader Angry (overall)",
        "Reader Sad (overall)",
        "Reader Anxious (overall)",
        "Reader Happy (overall)",
        "Author Angry (political)",
        "Author Sad (political)",
        "Author Anxious (political)",
        "Author Happy (political)",
        "Reader Angry (political)",
        "Reader Sad (political)",
        "Reader Anxious (political)",
        "Reader Happy (political)",
        "Partisanship",
        "Out-group Animosity",
        "In-group Perception",
        "Out-group Perception",
        "In-group Perception (left users)",
        "Out-group Perception (left users)",
        "In-group Perception (right users)",
        "Out-group Perception (right users)",
        "Reader Pref (overall)",
        "Reader Pref (political)",
    ]


def graph_effects_for_tl(
    eng_effects: pd.DataFrame,
    stated_effects: pd.DataFrame = None,
    stated_downranking_effects: pd.DataFrame = None,
    save_file=None,
):
    """
    Graphs the effects.

    Params:
        eng_effects: The effects for the engagement timeline.
        stated_effects: The effects for the stated timeline.
        stated_downranking_effects: The effects for the stated timeline with downranking.
        save_file: The file to save the graph to.
    """
    effects_dfs = [eng_effects, stated_effects, stated_downranking_effects]
    # Graph all effects, one effect per row
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 12))

    # Get y-axis labels and positions)
    y_labels = get_y_labels()
    y_positions = get_y_positions()

    # Plot the labels for test statistics with dummy point values
    sns.pointplot(
        x="Standardized Effect", y="Outcome", data=effects_dfs[0], join=False, ax=ax
    )
    sns.despine(top=True, right=True, bottom=True, left=True)
    sns.despine(top=True, right=True, bottom=True, left=True)
    ax.cla()  # Clear plot of dummy vals

    ax.set(ylabel=None)
    ax.set(xlabel="Effect Size (standard deviations)")
    ax.set_yticklabels(y_labels)
    ax.set_yticks(y_positions)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    # Mark 0 treatment effect
    ymin, ymax = ax.get_ylim()
    ax.vlines(
        x=0, ymin=ymin, ymax=ymax, color=sns.color_palette()[7], linestyle="solid"
    )

    # Names for each timeline for legend
    timeline_names = ["Engagement", "Stated Pref.", "SP-OA"]
    for effects_df, tl_name in zip(effects_dfs, timeline_names):
        if effects_df is None:
            continue
        effects_df = effects_df.sort_values(by="Outcome")

        if tl_name == "Engagement":
            color = sns.color_palette()[0]
        elif tl_name == "Stated Pref.":
            color = list(reversed(sns.color_palette("rocket")))[1]
        else:
            color = list(reversed(sns.color_palette("rocket")))[2]

        # Manually plot confidence intervals
        conf_ints = effects_df["Confidence Interval (standardized)"]
        cap_size = 0.25
        for (xmin, xmax), y_pos in zip(conf_ints, y_positions):
            ax.hlines(y=y_pos, xmin=xmin, xmax=xmax, colors=color, alpha=0.5)
            ax.vlines(
                x=xmin,
                ymin=y_pos - cap_size,
                ymax=y_pos + cap_size,
                colors=color,
                alpha=0.5,
            )
            ax.vlines(
                x=xmax,
                ymin=y_pos - cap_size,
                ymax=y_pos + cap_size,
                colors=color,
                alpha=0.5,
            )

        # Manually plot treatment effect data points
        treatment_effect_pts = effects_df["Standardized Effect"]
        for i, (t_effect, y_pos) in enumerate(zip(treatment_effect_pts, y_positions)):
            label = tl_name if i == 0 else None
            plt.plot(
                t_effect,
                y_pos,
                marker="o",
                color=color,
                mec=color,
                fillstyle="full",
                markerfacecolor="white",
                mew=2,
                label=label,
                alpha=0.75,
            )

    # Add labels for each group of effects
    ax.text(
        -0.55,
        0.82,
        "Emotions\n(all tweets)",
        transform=ax.transAxes,
        rotation=90,
        rotation_mode="anchor",
        horizontalalignment="center",
        weight="bold",
    )
    ax.text(
        -0.55,
        0.55,
        "Emotions\n(political tweets)",
        transform=ax.transAxes,
        rotation=90,
        rotation_mode="anchor",
        horizontalalignment="center",
        weight="bold",
    )
    ax.text(
        -0.55,
        0.28,
        "Political\nEffects",
        transform=ax.transAxes,
        rotation=90,
        rotation_mode="anchor",
        horizontalalignment="center",
        weight="bold",
    )
    ax.text(
        -0.55,
        0.1,
        "Reader\n Pref",
        transform=ax.transAxes,
        rotation=90,
        rotation_mode="anchor",
        horizontalalignment="center",
        weight="bold",
    )

    # Add legend
    ax.legend()

    # Save figure
    if save_file:
        plt.savefig(save_file, bbox_inches="tight")


def get_labels_and_title(key, labels):
    shortened_labels = {
        "user_race": {
            "American Indian or Alaska Native": "Native",
            "Asian": "Asian",
            "Black or African American": "Black",
            "Native Hawaiian or Pacific Islander": "Pacific Isl.",
            "White": "White",
            "Other": "Other",
        },
        "user_ethnicity": {
            "Spanish": "Spanish",
            "Hispanic": "Hispanic",
            "Latino": "Latino",
            "None of these": "None",
        },
        "user_annual_household_income": {
            "Less than $25,000": "< $25k",
            "$25,000 - $50,000": "$25 to $50k",
            "$50,000 - $100,000": "$50 to $100k",
            "$100,000 - $200,000": "$100 to $200k",
            "More than $200,000": "> $200k",
            "Prefer not to say": "Prefer not to say",
        },
        "user_age_group": {
            "18-24 years old": "18-24",
            "25-34 years old": "25-34",
            "35-44 years old": "35-44",
            "45-54 years old": "45-54",
            "55-64 years old": "55-64",
            "65-74 years old": "65-74",
            "75 years or older": "75+",
        },
        "user_education_level": {
            "Some high school": "Some HS",
            "High school graduate": "HS",
            "Associate degree": "Associate's",
            "Bachelor's degree": "Bachelor's",
            "Master's degree or above": "Master's",
            "Prefer not to answer": "Prefer not to say",
            "Other": "Other",
        },
        "user_gender": {
            "Woman": "Woman",
            "Man": "Man",
            "Non-binary": "Non-binary",
            "Other": "Other",
        },
        "user_main_why_tweet": {
            "A way to stay informed": "Info.",
            "Entertainment": "Entert.",
            "Keeping me connected to other people": "Connected",
            "It's useful for my job or school": "Job or school",
            "Lets me see different points of view": "PoVs",
            "A way to express my opinions": "Express opns",
        },
        "user_ideo_political_party": {
            "Republican": "Republican",
            "Democrat": "Democrat",
            "Independent": "Independent",
            "Something else": "Something else",
        },
        "user_ideo_political_leaning": {
            "Far left": "Far left",
            "Left": "Left",
            "Moderate": "Moderate",
            "Right": "Right",
            "Far right": "Far right",
            "Other": "Other",
        },
        "content_category": {
            "Entertainment": "Entertainment",
            "News": "News",
            "Hobbies": "Hobbies",
            "Politics": "Politics",
            "Work": "Work",
            "Other": "Other",
        },
    }

    title_d = {
        "user_race": "Race",
        "user_ethnicity": "Ethnicity",
        "user_annual_household_income": "Household Income",
        "user_age_group": "Age",
        "user_education_level": "Education Level",
        "user_gender": "Gender",
        "user_main_why_tweet": "Why Twitter",
        "user_ideo_political_party": "Political Party",
        "user_ideo_political_leaning": "Political Leaning",
        "content_category": "Primary Category of Content",
    }

    labels = [shortened_labels[key][l] for l in labels]
    title = title_d[key]
    return title, labels


def plot_heterogeneous(data, eng_effects, outcome):
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    fig.suptitle(f"Outcome variable: {outcome}", fontsize=24, weight="bold", y=1)

    # Shared y-label
    text_obj = fig.supylabel("CATE (stds)", fontsize=18, va="center")
    text_obj.set_position(
        (text_obj.get_position()[0] - 0.01, text_obj.get_position()[1])
    )

    # Compute global y-limits, taking all values and removing nans
    all_values = [
        val[1]
        for values in data.values()
        for val in values.values()
        if type(val[1]) == tuple
    ]
    all_values = [val for tup in all_values for val in tup]
    global_min = min(all_values) - 0.1
    global_max = max(all_values) + 0.1

    aggregate_effect = eng_effects[eng_effects["Outcome"] == outcome][
        "Standardized Effect"
    ].values[0]

    for ax, (grouping, values) in zip(axes.ravel(), data.items()):
        labels = list(values.keys())
        title, labels = get_labels_and_title(grouping, labels)
        nan_mask = [np.isnan(val[0]) for val in values.values()]
        labels = [label for label, mask in zip(labels, nan_mask) if not mask]
        effects = [val[0] for val, mask in zip(values.values(), nan_mask) if not mask]
        cis = [val[1] for val, mask in zip(values.values(), nan_mask) if not mask]
        lower, upper = zip(*cis)

        # Adjust x-axis limits for aesthetic spacing
        ax.set_xlim(-0.5, len(labels) - 0.5)

        # Highlight the y=0 line
        line1 = ax.axhline(
            0, color="gray", linestyle="--", linewidth=0.5, label="No Effect"
        )
        # Make the dashes bigger
        line1.set_dashes([10, 5])  # 10 points on, 5 points off

        # Highlight the y=aggregate_effect line
        line2 = ax.axhline(
            aggregate_effect,
            color="royalblue",
            linestyle="--",
            linewidth=0.5,
            label="ATE (stds)",
        )
        # Make the dashes bigger
        line2.set_dashes([10, 5])  # 10 points on, 5 points off

        for i, (effect, l, u) in enumerate(zip(effects, lower, upper)):
            color = "dimgrey"  # Consistent color for all effects

            # Plot confidence intervals
            cap_size = 0.05
            ax.vlines(x=i, ymin=l, ymax=u, colors=color, alpha=0.5)
            ax.hlines(
                y=l, xmin=i - cap_size, xmax=i + cap_size, colors=color, alpha=0.5
            )
            ax.hlines(
                y=u, xmin=i - cap_size, xmax=i + cap_size, colors=color, alpha=0.5
            )

            # Manually plot treatment effect data points using scatter
            # ax.scatter(i, effect, color='white', edgecolors=color, s=50, linewidths=2, alpha=0.75)
            ax.plot(
                i,
                effect,
                marker="o",
                color=color,
                mec=color,
                fillstyle="full",
                markerfacecolor="white",
                mew=2,
                alpha=0.75,
            )

        ax.set_title(title, fontsize=20)

        # Thicken the axes spines for subplot contours
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.grid(False)
        ax.set_ylim(global_min, global_max)  # Set global y-limits

        # Create 5 equally spaced ticks between ymin and ymax
        yticks = np.linspace(global_min, global_max, 5)
        # # Set y-tick positions
        yticks = np.round(yticks, 1)
        ax.set_yticks(yticks)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
        ax.set_ylabel("")  # Clear individual subplot y-labels

    fig.legend(handles=[line2], loc="upper right", bbox_to_anchor=(1, 1), fontsize=16)

    plt.tight_layout()
    plt.savefig(f"figures/het_effects/grouped_by_outcome/{outcome}.pdf", bbox_inches="tight")
    plt.show()


def plot_outcomes_for_grouping(data, aggregate_data, suptitle, outside_grouping):
    fig, axes = plt.subplots(6, 4, figsize=(20, 31))

    fig.suptitle(suptitle, fontsize=28, weight="bold", y=1)

    # Shared y-label
    fig.supylabel("CATE (stds)", fontsize=22, va="center")

    # Compute global y-limits, taking all values and removing nans
    all_values = [
        val[1]
        for values in data.values()
        for val in values.values()
        if type(val[1]) == tuple
    ]
    all_values = [val for tup in all_values for val in tup]
    global_min = min(all_values) - 0.1
    global_max = max(all_values) + 0.1

    for i, ax in enumerate(axes.ravel()):
        if i in [20, 23]:
            continue

        if i > 20:
            i -= 1
        if i >= len(data):
            break

        grouping, values = list(data.items())[i]
        labels = list(values.keys())
        _, labels = get_labels_and_title(outside_grouping, labels)
        nan_mask = [np.isnan(val[0]) for val in values.values()]
        labels = [label for label, mask in zip(labels, nan_mask) if not mask]
        effects = [val[0] for val, mask in zip(values.values(), nan_mask) if not mask]
        cis = [val[1] for val, mask in zip(values.values(), nan_mask) if not mask]
        lower, upper = zip(*cis)

        # Adjust x-axis limits for aesthetic spacing
        ax.set_xlim(-0.5, len(labels) - 0.5)

        # Highlight the y=0 line
        hline = ax.axhline(
            0, color="gray", linestyle="--", linewidth=0.5, label="No Effect"
        )
        hline.set_dashes([10, 5])  # 10 points on, 5 points off

        for i, (effect, l, u) in enumerate(zip(effects, lower, upper)):
            color = "dimgray"  # Consistent color for all effects

            # Plot confidence intervals
            cap_size = 0.05
            ax.vlines(x=i, ymin=l, ymax=u, colors=color, alpha=0.5)
            ax.hlines(
                y=l, xmin=i - cap_size, xmax=i + cap_size, colors=color, alpha=0.5
            )
            ax.hlines(
                y=u, xmin=i - cap_size, xmax=i + cap_size, colors=color, alpha=0.5
            )

            # Manually plot treatment effect data points using scatter
            # ax.scatter(i, effect, color='white', edgecolors=color, s=50, linewidths=2, alpha=0.75)
            ax.plot(
                i,
                effect,
                marker="o",
                color=color,
                mec=color,
                fillstyle="full",
                markerfacecolor="white",
                mew=2,
                alpha=0.75,
            )

            # Highlight the y=aggregate_effect line
            # print(aggregate_data[grouping])
            hline = ax.axhline(
                aggregate_data[grouping],
                color="royalblue",
                linestyle="--",
                linewidth=0.5,
                label="ATE (stds)",
            )
            hline.set_dashes([10, 5])  # 10 points on, 5 points off

        ax.set_title(grouping, fontsize=20)

        # Thicken the axes spines for subplot contours
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.grid(False)
        # ax.set_ylim(-1, 1)  # Set global y-limits

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("")  # Clear individual subplot y-labels

        # Create 5 equally spaced ticks between ymin and ymax
        yticks = np.linspace(min(lower) * 1.05, max(upper) * 1.05, 5)
        # # Set y-tick positions
        yticks = np.round(yticks, 1)
        ax.set_yticks(yticks)

    axes.ravel()[20].set_visible(False)
    axes.ravel()[23].set_visible(False)

    fig.legend(handles=[hline], loc="upper right", bbox_to_anchor=(1, 1), fontsize=20)

    plt.tight_layout()
    plt.savefig(
        f"figures/het_effects/grouped_by_group/{outside_grouping}.pdf", bbox_inches="tight"
    )
    plt.show()

def graph_effects_for_gpt(
    eng_effects: pd.DataFrame,
    gpt_eng_effects: pd.DataFrame,
    effects_indices: list,
    save_file=None,
):  
    # Graph all effects, one effect per row
    effects_dfs = [eng_effects, gpt_eng_effects]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))

    # Get y-axis labels and positions
    y_labels = [x for i, x in enumerate(get_y_labels()) if i in effects_indices]
    y_positions = [1, 2, 5, 6, 7, 8, 11, 12, 13, 14]

    # Plot the labels for test statistics with dummy point values
    sns.pointplot(
        x="Standardized Effect", y="Outcome", data=effects_dfs[0], join=False, ax=ax
    )
    sns.despine(top=True, right=True, bottom=True, left=True)
    sns.despine(top=True, right=True, bottom=True, left=True)
    ax.cla()  # Clear plot of dummy vals
    
    ax.set(ylabel=None)
    ax.set(xlabel="Effect Size (standard deviations)")
    ax.set_yticklabels(y_labels)
    ax.set_yticks(y_positions)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    # Mark 0 treatment effect
    ymin, ymax = min(y_positions), max(y_positions)
    ax.vlines(
        x=0, ymin=ymin, ymax=ymax, color=sns.color_palette()[7], linestyle="solid"
    )

    timeline_names = ["Engagement", "Engagement (GPT)"]
    for effects_df, tl_name in zip(effects_dfs, timeline_names):
        if effects_df is None:
            continue

        if tl_name == "Engagement":
            color = sns.color_palette()[0]
        else:
            color = list(reversed(sns.color_palette("rocket")))[3]

        # Manually plot confidence intervals
        conf_ints = effects_df["Confidence Interval (standardized)"]
        cap_size = 0.25
        for (xmin, xmax), y_pos in zip(conf_ints, y_positions):
            ax.hlines(y=y_pos, xmin=xmin, xmax=xmax, colors=color, alpha=0.5)
            ax.vlines(
                x=xmin,
                ymin=y_pos - cap_size,
                ymax=y_pos + cap_size,
                colors=color,
                alpha=0.5,
            )
            ax.vlines(
                x=xmax,
                ymin=y_pos - cap_size,
                ymax=y_pos + cap_size,
                colors=color,
                alpha=0.5,
            )

        # Manually plot treatment effect data points
        treatment_effect_pts = effects_df["Standardized Effect"]
        for i, (t_effect, y_pos) in enumerate(zip(treatment_effect_pts, y_positions)):
            label = tl_name if i == 0 else None
            plt.plot(
                t_effect,
                y_pos,
                marker="o",
                color=color,
                mec=color,
                fillstyle="full",
                markerfacecolor="white",
                mew=2,
                label=label,
                alpha=0.75,
            )

    # Add labels for each group of effects
    ax.text(
        -0.55,
        0.83,
        "Emotions\n(all tweets)",
        transform=ax.transAxes,
        rotation=90,
        rotation_mode="anchor",
        horizontalalignment="center",
        weight="bold",
    )
    ax.text(
        -0.55,
        0.43,
        "Emotions\n(political tweets)",
        transform=ax.transAxes,
        rotation=90,
        rotation_mode="anchor",
        horizontalalignment="center",
        weight="bold",
    )
    ax.text(
        -0.55,
        0.1,
        "Political\nEffects",
        transform=ax.transAxes,
        rotation=90,
        rotation_mode="anchor",
        horizontalalignment="center",
        weight="bold",
    )
    
    ax.legend()

    # Save figure
    if save_file:
        plt.savefig(save_file, bbox_inches="tight")