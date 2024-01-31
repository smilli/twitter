Code for the paper [Engagement, User Satisfaction, and the Amplification of Divisive Content on Social Media](https://arxiv.org/abs/2305.16941).

# Installation
1. To get started, create a Conda environment: `conda create -n twitter python=3.9.11`
2. Then, activate the Conda environment: `conda activate twitter`
3. Finally, install all requirements: `pip install -r requirements.txt`

# Data Access
To request access to our dataset, you can use [this form](https://forms.gle/mVtNTKMdi3XMARR1A).

# Data Description
Descriptions of the data files can be found in [`DATA.md`](DATA.md).

# Notebooks
- `effects.ipynb`: Average treatment effect (ATE) graph (Figure 1) and political tweets by in-group and out-group (Figure 2)
- `demographics.ipynb`: User demographics and ANES data comparison (SM section S2)
- `metadata_stats.ipynb`: Descriptive statistics for tweet metadata and individual user-level amplification (SM sections S4.1 & S4.2)
- `likert_distributions.ipynb`: Distribution of responses to Likert survey questions (SM section S4.3)
- `outcomes_by_rank.ipynb`: Outcomes broken down by the position of tweet in each timeline (SM section S4.4)
- `effects_by_tweet_threshold.ipynb`: ATE calculations across varied tweet thresholds (SM section S4.5)
- `gpt_judgements.ipynb`: ATEs for tweet outcomes calculated with GPT-4 labels (SM section S4.6)
- `effects_het.ipynb`: Effects calculated when restricting to subpopulations of users (SM section S4.7)

# Citing
If you use our code or data, please cite our paper:

```latex
@article{milli2023twitter,
  title={Engagement, User Satisfaction, and the Amplification of Divisive Content on Social Media},
  author={Milli, Smitha and Carroll, Micah and Wang, Yike and Pandey, Sashrika and Zhao, Sebastian and Dragan, Anca D.},
  journal={arXiv preprint arXiv:2305.16941},
  year={2023}
}
