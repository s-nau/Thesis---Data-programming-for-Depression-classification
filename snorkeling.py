# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 20:59:10 2023

@author: shimm
"""
#%%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import re 
#%%%

os.chdir("C:/Users/shimm/OneDrive - University of Toronto/Fourth Year/Courses/Thesis")

#cancer = pd.read_csv("cancer_rand.csv")
random = pd.read_csv("random_rand.csv")
dep = pd.read_csv("dep_rand.csv")


#%%%%
random["ds_label"] = "random"
#cancer["ds_label"] = "cancer" # need to justify, look back at that paper 
dep["ds_label"] = "depression"

#%%
data = pd.concat([random, #cancer,
                  dep])
#%%

# fig = plt.figure(figsize = (10, 5))
 
# # creating the bar plot
# plt.hist(data["upvote_ratio"].astype(str))
 
# plt.xlabel("upvote ratio")
# plt.ylabel("count")

# plt.show()

# #%%


 
# # creating the bar plot
# plt.bar(data["total_awards_received"].astype(str), height = 1)
 
# plt.xlabel("total_awards_received")
# plt.ylabel("count")

# plt.show()
# #%%
# plt.hist(data["num_comments"])
 
# plt.xlabel("num_comments")
# plt.ylabel("count")

# plt.show()

# #%%
# plt.hist(data["score"])
 
# plt.xlabel("score")
# plt.ylabel("count")

# plt.show()

# #%%

# data.groupby("upvote_ratio").count()
# # not meaningful
# #%%
# data.groupby("num_comments").count()

# # not meaningful
# #%%%

# data.groupby("total_awards_received").count()
# # not meaningful
# #%%
# data.groupby("score").count()
# #not mneaingful
# #%%
# data.groupby("view_count").count()
# # not meaningful

# #%%%
# data.groupby("is_video").count()
# # nto meaningful only 455 true

#%%%
data.drop(columns = ["upvote_ratio","num_comments","total_awards_received", "score", "view_count","is_video", "media_embed", "all_awardings", "awarders"],inplace=True)

#%%

def clean_first_go(x):
    x["selftext"] = x["selftext"].fillna("")
    
    
    lst = []
    
    for row in x["selftext"]:
        if row == "[removed]":
            lst.append("")
        else:
            lst.append(row)
                   
    x["selftext"]= lst
    
    
    lst2 = []
    
    for row in x["selftext"]:
        if row == "[deleted]":
            lst2.append("")
        else:
            lst2.append(row)
        
    x["selftext"]= lst2
    
    x["text"] = (x["title"] +" " + x["selftext"]).str.lower()
    # for some reason we continued having duplicates
    x.drop_duplicates(subset = ["text"], inplace = True)
    
    f = lambda x :re.sub('&amp', "and",x)
    x["text"] = pd.Series(x["text"]).apply(f)  
    
    import string
    
    
    
    f_2 = lambda x: bool(all(j.isdigit() or j in string.punctuation for j in x))
    f_2_t = pd.Series(x["text"]).apply(f_2)  
    
    x["r"] = f_2_t
    x = x[x["r"] != True]
    
    x.drop(columns = ["r"], inplace = True)
    
    rex = r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)"
    f_3 = lambda x: re.sub(rex, "" ,x)
    x["text"] = x["text"].apply(f_3)

clean_first_go(data)


#%%%
# read out test data
#data_2 = pd.concat([random, cancer, dep])
#data_2.drop(columns = ["upvote_ratio","num_comments","total_awards_received", "score", "view_count","is_video", "media_embed", "all_awardings", "awarders"],inplace=True)
#clean_first_go(data_2)
#test_to_label = data_2.sample(n=1000, random_state = 1)

#test_to_label.to_csv("to_label.csv")

#%%%
#read back in test data

data_test = pd.read_csv("to_label - results.csv")

#%%%
data_test = data_test[data_test["miriam's label" ].notnull()]

data_test["label"] = np.where(data_test["miriam's label"] == "yes", 1, 0)

data_test = data_test[data_test["ds_label"] != "cancer"]

test_ind = list(data_test["Unnamed: 0.2"])

#%%
data['index_1'] = data.index
data = data[~data.index_1.isin(test_ind)]

data.drop(columns = ["index_1"], inplace = True)
#%%

import textblob as tb 
# import text2emotion as te
import sklearn as skl
import nltk
#%%%
# ends up with 40 to labeler set
# train, labeller = skl.model_selection.train_test_split(data,random_state = 496, train_size = .999, test_size = .001, stratify =data["ds_label"])

#%%%

import snorkel
#%%
# from LeXmo import LeXmo # no longer use
from nrclex import NRCLex 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#%%
# =============================================================================
# #developing label functions
# lst = []
# for index, row in labeller.iterrows():
#     emo=LeXmo.LeXmo(row["text"])
#     emo.pop('text', None)
#     lst.append((emo))
# 
# p_1 = pd.DataFrame(lst)
# p_1["index_l"] = labeller.index
# 
# p_1 = p_1.merge(labeller, left_on = "index_l", right_index = True)
# 
# dist = p_1.groupby("ds_label").mean("anger")
# 
# lst_2 = []
# 
# for index, row in labeller.iterrows():
#     t=tb.TextBlob(row["text"])
#     p = t.sentiment
#     lst_2.append((p.polarity, p.subjectivity , index))
#     
# p = pd.DataFrame(lst_2)
# p.columns = ["polarity", "subjectivity", "index_l"]
# 
# p = p.merge(labeller, left_on = "index_l", right_index = True)
# 
# 
# 
# p.groupby("ds_label").mean("polarity")
# 
# 
# 
# t = tb.TextBlob(labeller.iloc[1]["text"])
# pos = t.tags
# p = pd.DataFrame(pos)
# p.columns = ["word", "Pos"]
# =============================================================================

#%%
from snorkel.labeling import labeling_function
#%%
import emojis
#%%
Depression = 1
Abstain = -1
Not_depression = 0

@labeling_function()
def emotion_dep_LF(x):
    t = x["text"]
    emot=NRCLex(t)
    top = emot.top_emotions
    s = [x for x in top if x[0] == "sadness"]
    # if len(s) > 0:
    if len(s) > 0 and s[0][0] == "sadness" and s[0][1] > 0:
        return Depression
    else:
        return Abstain
        
@labeling_function()
def subreddit_Lf(x):
    rex = r'\b(depression)\b'
    sub = str.lower(x["subreddit"])
    if bool(re.match(rex, sub)):
        return Depression
    else:
        return Abstain

@labeling_function()
def emotion_n_dep_Lf(x):
    t = x["text"]
    emot=NRCLex(t)
    top = emot.top_emotions
    j = [x for x in top if x[0] == "joy"]
    if len(j) > 0  and j[0][0] == "joy" and j[0][1] > 0:
        return Not_depression
    else:
        return Abstain
    
@labeling_function()        
def sent_Lf_dep(x):
    """
    
    Info from here 
    https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    The compound score is computed by summing the valence scores
    of each word in the lexicon, adjusted according to the rules,
    and then normalized to be between -1 (most extreme negative)
    and +1 (most extreme positive)

    """
    s = SentimentIntensityAnalyzer()
    d = s.polarity_scores(x["text"])["compound"] 
    if d < -0.90:
        return Depression
    else:
        return Abstain
 
@labeling_function()        
def sent_Lf_ndep(x):
    s = SentimentIntensityAnalyzer()
    d = s.polarity_scores(x["text"])["compound"] 
    if d > 0.90:
        return Not_depression
    else:
        return Abstain
    # t = tb.TextBlob(x["text"])
    # if t.sentiment.polarity < 0:
    #     return Depression
    # else:
    #     Abstain

@labeling_function()
def personal_pronouns_Lf(x):
    """
    # from https://www.thesaurus.com/e/grammar/first-person-pronouns/ and 
    #(Ramirez-Esparza et al. 2021)

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    t = tb.TextBlob(x["text"])
    w = t.word_counts
    fps = ["I", "me", "myself", "mine"]
    fpp = ["we", "ours", "ourselves", "us"]
    fps_count = np.sum([w.get(k, 0) for k in fps])
    fpp_count = np.sum([w.get(t, 0) for t in fpp])
    if fps_count > fpp_count:
        return Depression
    else:
        return Abstain

    # pos = t.tags
    # p = pd.DataFrame(pos)
    # p.columns = ["word", "Pos"]
    # p_f = p[p["Pos"] == "PRP"]
    # if p_f.shape[0] > len(nltk.sent_tokenize(x["text"])):
    #     return Depression
    # else:
    #     return Abstain

@labeling_function()
def dep_word_Lf(x):
    if bool(re.match(r'.*(i|me|myself).*depr.*', x["text"])):
        return Depression
    else:
        return Abstain
    
@labeling_function() 
def word_Lf(x):
    """
    # “depressed”, “suffer”, “attempt”, “suicide”, “battle”,
    # “struggle”, “diagnosed”, in addition to first person pronouns, (Jamil et al. 2017)
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if bool(re.match(r'.*(i|me|myself).*(depressed|suffer|attempt|suicide|battle|struggle|diagnosed).*', x["text"])):
        return Depression
    else:
        return Abstain
 
    
@labeling_function()
def word2_Lf(x):
    """
    # (Shen et al. 2013)
    # lonely, out of breath
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if bool(re.match(r'.*(lonely|out of breath).*', x["text"])):
        return Depression
    else:
        return Abstain


@labeling_function()
def drug_Lf(x):
    """
    # mention of depression drugs (Vedula and Parthasarathy 2017)
    drug list from 
    https://www.camh.ca/en/health-info/mental-illness-and-addiction-index/antidepressant-medications
    # brand names can be different in different countries
    # for usa gotten from https://www.mayoclinic.org/drugs-supplements/paroxetine-oral-route/description/drg-20067632
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # get european names
    
    ssri =  ["fluoxetine",
             "Prozac", 
             "paroxetine",
             "paxil",
             "pexeva",
             "brisedelle"
             "fluvoxamine",
             "Luvox", 
             "citalopram",
             "Celexa",
             "escitalopram",
             "lexapro"
             "cipralex",
             "sertraline", 
             "Zoloft"]
    snri = ["venlafaxine",
            "Effexor",
            "duloxetine",
            "Cymbalta",
            "irenka",
            "levomilnacipran",
            "Fetzima",
            "desvenlafaxine",
            "Pristiq",
            "khedezla"]
    NDRI =[ "bupropion", 
           "Wellbutrin",
           "aplensin",
           "budeprion",
           "buproban",
           "forfivo",
           "Zyban"]   
    NaSSAs = ["Mirtazapine" "Remeron"]
    Nonselective_cyclics = ["amitriptyline",
                            "Elavil",
                            "vanatrip",
                            "imipramine",
                            "Tofranil",
                            "desipramine"
                            "Norpramin", 
                            "nortriptyline",
                            "Aventyl", 
                            "pamelor",
                            "trimipramine",
                            "Surmontil",
                            "clomipramine",
                            "Anafranil"]
    
    MAOIs = ["phenelzine",
             "Nardil", 
             "tranylcypromine",
             "Parnate"]
    
    drugs = ssri + snri + NDRI + NaSSAs + Nonselective_cyclics + MAOIs
    drugs = [d.lower() for d in drugs]  
    rex = r"(.*("+'|'.join(drugs)+r".*))"
    if bool(re.match(rex, x["text"])):
        return Depression
    else:
        return Abstain


# @labeling_function()
# def relationship_Lf(x):
#     """
#     Help from chatgpt with regex

#     Parameters
#     ----------
#     x : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     d = x["text"]
#     rex = r"(?<!\bnot\s)(?<!\bnever\s)(?<!\bdon't\s)(?<!\bdoesn't\s)(?<!\baren't\s)(?<!\bisn't\s)(?<!\bwasn't\s)(?<!\bwon't\s)(?<!\bwouldn't\s)(?<!\bcan't\s)(?<!\bcouldn't\s)(?<!\bain't\s)(boyfriend|girlfriend|partner|spouse|husband|wife|significant\s*other|love\s*interest|romantic\s*interest|romantic\s*partner|romantic\s*relationship|dating|bride|groom)"
#     if bool(re.match(rex, d)):
#         return Not_depression
#     else:
#         return Abstain
    
    
    
# @labeling_function()
# def Q_mark_lf(x):
#     d = x["text"]
#     sentences = [nltk.word_tokenize(t) for t in 
#      nltk.sent_tokenize(d)]
#     rex = r'\?'
#     qmarks = re.findall(rex, d)

#     if len(qmarks) > len(sentences)//2:
#         return Depression
#     else:
#         return Abstain
    
@labeling_function()
def len_lf(x):
    d = x["text"]
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(d)
    if len(words) <= 4:
        return Not_depression
    else:
        return Abstain
    
    
    
# @labeling_function()
# def meds_Lf(x):
#     if bool(re.match(r".*meds.*", x["text"])):
#         return Depression
#     else:
#         return Abstain
    
@labeling_function()
def despair_Lf(x):
    """
    help from chat GPT

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    rex = r"\b(hopeless|helpless|worthless|lonely|isolated|despair|desperate|pointless|meaningless|empty|lost|depressed|depression|suicidal|suicide|dying|endless|futile|bleak)\b"
    if bool(re.match(rex, x["text"])):
        return Depression
    else:
        return Abstain
    
@labeling_function()
def emojis_Lf(x): 
    """
    from (Stephen and P 2019)

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    lst_of_em = emojis.get(x["text"])
    if len(lst_of_em) > 0:
        return Not_depression
    else:
        return Abstain
    
    
@labeling_function()
def postitive_words_Lf(x):
    """
    used chatgpt

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    rex = r'\b(?<!not\s)(happy|excited|grateful|joy|pleasure|content|blissful|ecstatic|elated|euphoric|overjoyed|radiant|thrilled|gleeful|thankful|appreciative|admiration|enthusiastic)\b'
    d = x["text"]
    if bool(re.match(rex, d)):
        return Not_depression
    else:
        return Abstain
    
# @labeling_function()
# def going_out_Lf(x):
#     """
#     used chatgpt
    
    
#     adjust to increase the capture

#     Parameters
#     ----------
#     x : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     rex = r'\b((went (out|to|on)|had|did|went|planned) (something )?(fun|exciting|relaxing|interesting|enjoyable|outdoor|indoor|social|cultural) (activity|event|outing|experience|time) (with|together with))\s+([\w\s]+)\b'
#     d = x["text"]
#     if bool(re.match(rex, d)):
#         return Not_depression
#     else:
#         return Abstain
    
@labeling_function()
def neg_cog_Lf(x):
    """
    used chatpgt

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    rex = r'\b(nothing is worth it|what\'s the point|who cares|not worth it|not important|pointless|hopeless|meaningless|meaningless)\b'
    d = x["text"]
    if bool(re.match(rex, d)):
        return Depression
    else:
        return Abstain
@labeling_function()
def skip_work_Lf(x):
    """
    chatgpt

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    rex = r'\b(not going into|not going to|skipping|taking a break from|calling off|staying home from|not attending|not showing up for|missing) (work|school)\b'
    d = x["text"]
    if bool(re.match(rex, d)):
        return Depression
    else:
        return Abstain
    
@labeling_function()

def physical_state_Lf(x):
    """
    chatgpt

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    rex = r'\b(can\'t (?:get out of bed|remember)|pounding|aching|unwell|curl up in a ball|remember when I last ate)\b'
    d = x["text"]
    if bool(re.match(rex, d)):
        return Depression
    else:
        return Abstain
# @labeling_function()
# def links_Lf(x):
#     """
#     need a source 
#     """
#     return 
    # def dep_word_Lf(x):
#     if bool(re.match(r'.*happy.*', x["text"])) and not  bool(re.match(r'.*not happy.*', x["text"])):
#         return Not_depression
#     else:
#         Abstain
        
# (Shen et al. 2013)
# lonely, out of breath
#


# mention of depression drugs (Vedula and Parthasarathy 2017)

# “depressed”, “suffer”, “attempt”, “suicide”, “battle”,
 # “struggle”, “diagnosed”, in addition to first person pronouns, (Jamil et al. 2017)
 
 # regex stuff (Prieto et al. 2014)
 
 # more first person personal pronouns (Ramirez-Esparza et al. 2021)
 
 # Negatively valenced words (Pirina and Çöltekin 2018)
 
 
 #%%%
 # negative settesting stuff
rex2 =  r"(?!.*\b(not|un) diagnosed\b)(?=.*\b(i|me|myself)\b)(?=.*\bdiagnosed\b)(?=.*\bdepression\b).*$"
# written by chatgpt
# true labels

z =  data[data.text.str.contains(rex2)]




z.to_csv("self_dec.csv")

#%%
# positive set
z_pos = data[data["subreddit"] != "depression"].sample(random_state = 1, n = z.shape[0], replace = False)
rex_3 = r'\b(anxiety|anxious|autism|autistic|bipolar|borderline|depression|depressed|dysthymia|eating disorder|grief|mania|manic|mental illness|obsessive compulsive disorder|panic attack|panic disorder|paranoia|personality disorder|phobia|post-traumatic stress disorder|psychotic|schizophrenia|self-harm|social anxiety|suicidal|suicide|trauma|unipolar|depressive|behavioral disorder|conduct disorder|oppositional defiant disorder|substance abuse|alcoholism|drug addiction|addiction|dependence|withdrawal|trauma|stress|burnout|fatigue|exhaustion|sleep disorder|insomnia|nightmares|flashbacks|hallucinations|delusions|psychosis|psychosomatic|hypochondria)\b'
# by chatgpt
z_pos_test = z_pos[z_pos.text.str.contains(rex_3)]["author"]

z_pos = z_pos[~z_pos["author"].isin(set(z_pos_test))]

z_pos.to_csv("not_self_dec.csv")


#%%
#look at misclasfficiation
#%%


test_post = pd.read_csv("self_dec_scp.csv",dtype = {'title': 'O',
 'selftext': 'O',
 'subreddit': 'O',
 'author': 'O',
 'author_fullname': 'O',
 'created_utc': 'int64',
 'utc_datetime_str': 'O'})

test_post_neg =  pd.read_csv("not_self_dec_scp.csv",dtype = {'title': 'O',
 'selftext': 'O',
 'subreddit': 'O',
 'author': 'O',
 'author_fullname': 'O',
 'created_utc': 'int64',
 'utc_datetime_str': 'O'})
clean_first_go(test_post)
clean_first_go(test_post_neg)
test_post.drop(columns = ["created_utc" , 'Unnamed: 0', "r"], inplace = True)


test_post_neg.drop(columns = ["created_utc" , 'Unnamed: 0', "r"], inplace = True)

test_post_train, test_post_test  = skl.model_selection.train_test_split(test_post, test_size = .5, random_state = 1)
test_post_neg_train, test_post_neg_test = skl.model_selection.train_test_split(test_post_neg, test_size = .5, random_state = 1)
to_cond = pd.concat([test_post_test['text'],test_post_train['text']])
cond = data['text'].isin(to_cond)
data.drop(data[cond].index, inplace = True)

to_cond2 = pd.concat([test_post_neg_test['text'],test_post_neg_train['text']])
cond2 = data['text'].isin(to_cond2)
data.drop(data[cond2].index, inplace = True)


test_post_test["label"] = Depression
test_post_neg_test["label"] = Not_depression

test_set = pd.concat([test_post_test, test_post_neg_test])

# 'Unnamed: 0.1', 'Unnamed: 0', 'subreddit_name_prefixed',  'author', 'retrieved_utc', 'updated_utc', 'utc_datetime_str',
       # 'lang', 'ds_label', 'text', 'r'
data.drop(columns = ["Unnamed: 0"], inplace = True)
data.drop(columns = ["subreddit_name_prefixed","author_premium","edited","retrieved_utc","updated_utc","lang","ds_label", "r"], inplace = True)    

data = pd.concat([data, test_post_train, test_post_neg_train])

data.drop(columns = ["Unnamed: 0.1"], inplace = True)


#%%



#%%
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

#%%

lfs = [emotion_dep_LF, subreddit_Lf, 
       emotion_n_dep_Lf, sent_Lf_dep,
       sent_Lf_ndep,
       personal_pronouns_Lf,
       dep_word_Lf, word_Lf,
       word2_Lf, drug_Lf,
       # relationship_Lf,
       # Q_mark_lf,
       len_lf,
       # meds_Lf,
       despair_Lf,
       emojis_Lf,
       postitive_words_Lf,
       # going_out_Lf,
       neg_cog_Lf,
       skip_work_Lf,
       physical_state_Lf
       ]


#%%

# test_set, lf_val_set = skl.model_selection.train_test_split(test_set, random_state = 1, stratify = test_set["label"])
#%%

# applier = PandasLFApplier(lfs)
# L_train = applier.apply(lf_val_set)

# lfa = LFAnalysis(L=L_train, lfs=lfs)

# print(lfa.lf_summary())

#%%

# label_model = LabelModel()
# label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
# z_3 = label_model.predict_proba(L_train)

# z_2 = np.where(z_3[:, 1] > .5, 1, 0 )

# lf_val_set["eval_labels"] = z_2

# print(lfa.lf_empirical_accuracies(lf_val_set["label"]))

# print(np.mean(np.where(lf_val_set["label"] == lf_val_set["eval_labels"],1,0)))

#%%
applier = PandasLFApplier(lfs)
L_train = applier.apply(data)

lfa = LFAnalysis(L=L_train, lfs=lfs)

lfa_sum = lfa.lf_summary()

print(lfa_sum)

#%%%
# cov_emotion_dep_LF, cov_subreddit_Lf,  cov_emotion_n_dep_Lf, cov_sent_Lf, cov_personal_pronouns_Lf, cov_dep_word_Lf, cov_word_Lf, cov_word2_Lf, cov_drug_Lf= (L_train != Abstain).mean(axis=0)

# print((cov_emotion_dep_LF, cov_subreddit_Lf,  cov_emotion_n_dep_Lf, cov_sent_Lf, cov_personal_pronouns_Lf, cov_dep_word_Lf, cov_word_Lf, cov_word2_Lf, cov_drug_Lf))


#%%%


label_model = LabelModel()
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)

z_3 = label_model.predict_proba(L_train)

z_2 = np.where(z_3[:, 1] > .5, 1, 0 )




blnc = ["depr",np.mean(np.where(z_2 == Depression, 1, 0)), "not_depr",np.mean(np.where(z_2 == Not_depression, 1, 0))]

blnc_data_forums = ["depr", np.mean(np.where(data["subreddit"] == "depression", 1 ,0)), "not depr", np.mean(np.where(data["subreddit"]!= "depression" , 1 , 0))]
## problem with class imbalance
#%%%
data["label"] = z_2
#%%

from nltk.corpus import stopwords

# remove the stop words
def stopword_clean(df):
    sw =  stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))
print(data.shape[0])
stopword_clean(data)
print(data.shape[0])
print(data_test.shape[0])
stopword_clean(data_test)
print(data_test.shape[0])
#%%
# BOW
# from sklearn.feature_extraction.text import CountVectorizer

# v = CountVectorizer()


# z_sp_2 = pd.DataFrame.sparse.from_spmatrix(v.fit_transform(data["text"].to_numpy()))

# zs = z_sp_2.astype(bool).sum(axis=0)

# zs = [i for i ,j in zs.iteritems() if j == 1]





# z_sp_2.drop(columns = zs, axis = 1, inplace = True)



# data["w_id"] = range(data.shape[0])
# data.set_index(["w_id"], inplace = True)

# data = pd.merge(data, z_sp_2, left_index = True, right_index = True)
#%%
#Emotion
def emot_add(x): 
    z_emot = x["text"].apply(lambda x: NRCLex(x).raw_emotion_scores)
    emots = pd.json_normalize(pd.DataFrame(z_emot)["text"])
    
    emots.fillna(0, inplace = True)
    x["wid"] = range(emots.shape[0])
    x.set_index(["wid"], inplace = True)
    x = pd.merge(x, emots, left_index = True, right_index = True)
    return x
data = emot_add(data)
data_test= emot_add(data_test)
#%%%
#polarity
def pol(df):
    s = SentimentIntensityAnalyzer()

    df["sent"] = df["text"].apply(lambda x: s.polarity_scores(x)["compound"] )



pol(data)
pol(data_test)
#%%%

#data["label"] = z_2

#data = data[data["label"] != Abstain]
#fix this drop
#data.drop(columns = data.columns[[0,1,3,4,6,7,8,9,10,11,13,14]],axis = 1, inplace = True)

#%%%
# Something is going on with the data here. Need to fix.
#SVM stuff

def tok(x):
    import collections
    
    np.random.seed(1)
    
    
    
    x["tok"] =np.array([nltk.tokenize.word_tokenize(t["text"]) for i, t in x.iterrows()])
    
    tag_map = collections.defaultdict(lambda : nltk.corpus.wordnet.NOUN)
    tag_map['J'] = nltk.corpus.wordnet.ADJ
    tag_map['V'] = nltk.corpus.wordnet.VERB
    tag_map['R'] = nltk.corpus.wordnet.ADV



    for index,entry in x["tok"].items():
    # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
    # Initializing WordNetLemmatizer()
        word_Lemmatized = nltk.stem.WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in nltk.pos_tag(entry):
    # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
        x.loc[index,'text_final'] = str(Final_words)

tok(data)
tok(data_test)
#%%
#POs stuff
def pos(df):
    tokzer = nltk.RegexpTokenizer(r'\w+')
    z_pos = df["text"].apply(lambda x: nltk.pos_tag(tokzer.tokenize(x)))
    f_pos = lambda x: [z[1] for z in x]
    
    z_pos = z_pos.apply(f_pos)
    
    from collections import Counter
    
    
    f_cont = lambda x: dict(Counter(x))
    
    z_pos = z_pos.apply(f_cont)
    
    def f_cont_norm(x):
        v = np.sum(list(x.values()))
        if v > 0:
             for k in x.keys(): 
                 x[k] = x[k]/v
    
    z_pos.apply(lambda x : f_cont_norm(x))
    
    z_pos_df = z_pos.apply(pd.Series)
    z_pos_df.fillna(0, inplace = True)
    df["wid"] = range(z_pos_df.shape[0])
    df.set_index(["wid"], inplace = True)

    df = pd.merge(z_pos_df, df, left_index = True, right_index = True)
    return df

data = pos(data)
data_test = pos(data_test)
#%%%
# need a better way of getting the desired features
Train_X = data.drop(columns = ["label"])
Train_Y = data["label"]
Test_X = data_test.drop(columns = ["label"])
Test_Y = data_test["label"]
Tfidf_vect = skl.feature_extraction.text.TfidfVectorizer(max_features=5000)

tf_trainer = pd.concat([Train_X["text_final"], Test_X["text_final"]])
Tfidf_vect.fit(tf_trainer)
#%%%
Train_X_Tfidf = Tfidf_vect.transform(Train_X["text_final"])
Test_X_Tfidf = Tfidf_vect.transform(Test_X["text_final"])

#%%
df_tidf_train = pd.DataFrame.sparse.from_spmatrix(Train_X_Tfidf)
Train_X["wid"] = range(df_tidf_train.shape[0])
Train_X.set_index(["wid"], inplace = True)
Train_X  = pd.merge(df_tidf_train, Train_X, left_index = True, right_index = True)
df_tidf_test= pd.DataFrame.sparse.from_spmatrix(Test_X_Tfidf)
Test_X["wid"] = range(df_tidf_test.shape[0])
Test_X.set_index(["wid"], inplace = True)
Test_X = pd.merge(df_tidf_test, Test_X, right_index= True, left_index= True)
#%%%
Test_X.drop(columns = [
     'Unnamed: 0.2',            'Unnamed: 0.1',
                    'Unnamed: 0',               'subreddit',
                      'selftext',         'author_fullname',
                         'title', 'subreddit_name_prefixed',
                'author_premium',                  'edited',
                        'author',           'retrieved_utc',
                   'updated_utc',        'utc_datetime_str',
                          'lang',                'ds_label',
                          'text',                       'r',
                "miriam's label",                    
                           'tok',              'text_final'
    # "text_final",
    #                    "subreddit", 
    #                    "title", 
    #                    "author", 
    #                    "utc_datetime_str",
    #                    "tok", 
    #                    "selftext",
    #                    "author_fullname",
    #                    "text"
                       ],
            inplace = True)
Train_X.drop(columns = [
    'Unnamed: 0.1', 'Unnamed: 0',
       'subreddit', 'selftext', 'author_fullname', 'title',
       'subreddit_name_prefixed', 'author_premium', 'edited', 'author',
       'retrieved_utc', 'updated_utc', 'utc_datetime_str', 'lang', 'ds_label',
       'text', 'r','tok',
       'text_final'
    # "text_final",
    #                    "subreddit", 
    #                    "title", 
    #                    "author", 
    #                    "utc_datetime_str",
    #                    "tok", 
    #                    "selftext",
    #                    "author_fullname",
    #                    "text"
                       ], inplace = True)

#%%
Train_X.columns = Train_X.columns.astype(str) 
Test_X.columns = Test_X.columns.astype(str) 
#%%
SVM = skl.svm.SVC(kernel='linear', class_weight="balanced")



SVM.fit(Train_X,Train_Y)

t_cols = list(Train_X.columns)

Test_X["''"] = 0
Test_X["NNPS"] = 0
Test_X["PDT"] = 0
Test_X["SYM"] = 0
Test_X["$"] = 0
Test_X["POS"] = 0
Test_X["RBS"] = 0
Test_X["WP$"] = 0



Test_X = Test_X[t_cols] # needed to reorder the columns to match

predictions_SVM = SVM.predict(Test_X)
print(skl.metrics.accuracy_score(predictions_SVM, Test_Y))
print(skl.metrics.f1_score(predictions_SVM, Test_Y))
print(skl.metrics.roc_auc_score(predictions_SVM, Test_Y))

"""
#run number two
print(skl.metrics.accuracy_score(predictions_SVM, Test_Y))
0.8305084745762712

print(skl.metrics.f1_score(predictions_SVM, Test_Y))
0.8484848484848484

print(skl.metrics.roc_auc_score(predictions_SVM, Test_Y))
0.868421052631579
"""

#new LFS

# 0.6962025316455697
# 0.6230366492146596
# 0.7198710568731291


#%%%
Train_Y = np.where(data["subreddit"] == "depression", 1, 0)
Train_X = data.drop(columns = ["label"])
# Train_Y = data["label"]
Test_X = data_test.drop(columns = ["label"])
Test_Y = data_test["label"]
Tfidf_vect = skl.feature_extraction.text.TfidfVectorizer(max_features=5000)

tf_trainer = pd.concat([Train_X["text_final"], Test_X["text_final"]])
Tfidf_vect.fit(tf_trainer)

#%%%%
Train_X_Tfidf = Tfidf_vect.transform(Train_X["text_final"])
Test_X_Tfidf = Tfidf_vect.transform(Test_X["text_final"])


#%%%%

df_tidf_train = pd.DataFrame.sparse.from_spmatrix(Train_X_Tfidf)
Train_X["wid"] = range(df_tidf_train.shape[0])
Train_X.set_index(["wid"], inplace = True)
Train_X  = pd.merge(df_tidf_train, Train_X, left_index = True, right_index = True)
df_tidf_test= pd.DataFrame.sparse.from_spmatrix(Test_X_Tfidf)
Test_X["wid"] = range(df_tidf_test.shape[0])
Test_X.set_index(["wid"], inplace = True)
Test_X = pd.merge(df_tidf_test, Test_X, right_index= True, left_index= True)



#%%%%

# Test_X.drop(columns = ["text_final",
#                        "subreddit", 
#                        "title", 
#                        "author", 
#                        "utc_datetime_str",
#                        "tok", 
#                        "selftext",
#                        "author_fullname",
#                        "text"],
#             inplace = True)
# Train_X.drop(columns = ["text_final",
#                        "subreddit", 
#                        "title", 
#                        "author", 
#                        "utc_datetime_str",
#                        "tok", 
#                        "selftext",
#                        "author_fullname",
#                        "text"], inplace = True)

#%%%%
Test_X.drop(columns = [
     'Unnamed: 0.2',            'Unnamed: 0.1',
                    'Unnamed: 0',               'subreddit',
                      'selftext',         'author_fullname',
                         'title', 'subreddit_name_prefixed',
                'author_premium',                  'edited',
                        'author',           'retrieved_utc',
                   'updated_utc',        'utc_datetime_str',
                          'lang',                'ds_label',
                          'text',                       'r',
                "miriam's label",                    
                           'tok',              'text_final'
    # "text_final",
    #                    "subreddit", 
    #                    "title", 
    #                    "author", 
    #                    "utc_datetime_str",
    #                    "tok", 
    #                    "selftext",
    #                    "author_fullname",
    #                    "text"
                       ],
            inplace = True)
Train_X.drop(columns = [
    'Unnamed: 0.1', 'Unnamed: 0',
       'subreddit', 'selftext', 'author_fullname', 'title',
       'subreddit_name_prefixed', 'author_premium', 'edited', 'author',
       'retrieved_utc', 'updated_utc', 'utc_datetime_str', 'lang', 'ds_label',
       'text', 'r','tok',
       'text_final'
    # "text_final",
    #                    "subreddit", 
    #                    "title", 
    #                    "author", 
    #                    "utc_datetime_str",
    #                    "tok", 
    #                    "selftext",
    #                    "author_fullname",
    #                    "text"
                       ], inplace = True)

#%%%%
Train_X.columns = Train_X.columns.astype(str) 
Test_X.columns = Test_X.columns.astype(str) 
#%%%%
SVM = skl.svm.SVC(kernel='linear', class_weight="balanced")



SVM.fit(Train_X,Train_Y)

t_cols = list(Train_X.columns)

Test_X["''"] = 0
Test_X["NNPS"] = 0
Test_X["PDT"] = 0
Test_X["SYM"] = 0
Test_X["$"] = 0
Test_X["POS"] = 0
Test_X["RBS"] = 0
Test_X["WP$"] = 0

Test_X = Test_X[t_cols] # needed to reorder the columns to match

predictions_SVM = SVM.predict(Test_X)
print(skl.metrics.accuracy_score(predictions_SVM, Test_Y))
print(skl.metrics.f1_score(predictions_SVM, Test_Y))
print(skl.metrics.roc_auc_score(predictions_SVM, Test_Y))

"""
print(skl.metrics.accuracy_score(predictions_SVM, Test_Y))
0.8305084745762712

print(skl.metrics.f1_score(predictions_SVM, Test_Y))
0.8437499999999999

print(skl.metrics.roc_auc_score(predictions_SVM, Test_Y))
0.8532608695652174
"""


