# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext && \
# rm e.l && python demo_scaled_f_score.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
from scattertext import SampleCorpora, whitespace_nlp_with_sentences, produce_frequency_explorer
from scattertext.CorpusFromPandas import CorpusFromPandas
from scattertext.termscoring.ScaledFScore import ScaledFScorePresetsNeg1To1

import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns',None)
from Data.utils import utils_data as utils_data

# ================================================================================

# convention_df = SampleCorpora.ConventionData2012.get_data()
# corpus = CorpusFromPandas(convention_df,
#                           category_col='party',
#                           text_col='text',
#                           nlp=whitespace_nlp_with_sentences).build().get_unigram_corpus()
# html = produce_frequency_explorer(corpus,
#                                   category='democrat',
#                                   category_name='Democratic',
#                                   not_category_name='Republican',
#                                   minimum_term_frequency=5,
#                                   width_in_pixels=1000,
#                                   term_scorer=ScaledFScorePresetsNeg1To1(
# 	                                      beta=1,
# 	                                      scaler_algo='normcdf'
#                                       ),
#                                   grey_threshold=0,
#                                   y_axis_values=[-1, 0, 1],
#                                   metadata=convention_df['speaker'])
# fn = './demo_scaled_f_score.html'
# open(fn, 'wb').write(html.encode('utf-8'))
# print('Open ' + fn + ' in Chrome or Firefox.')

# ================================================================================
all_satisfaction_score_comment_in_all_conds=utils_data.get_all_satisfaction_score_comment_in_all_conds()

columns=['senti_on_Metfor_oral','feature','review']
all_satisfaction_score_comment_in_all_conds_df=pd.DataFrame(all_satisfaction_score_comment_in_all_conds,index=None,columns=columns)
# print("all_satisfaction_score_comment_in_all_conds_df",all_satisfaction_score_comment_in_all_conds_df)

# ================================================================================
corpus=CorpusFromPandas(
    all_satisfaction_score_comment_in_all_conds_df,category_col='senti_on_Metfor_oral',text_col='review',
    nlp=whitespace_nlp_with_sentences).build().get_unigram_corpus()

# ================================================================================
html=produce_frequency_explorer(
    corpus,category='negative',category_name='Negative',not_category_name='Positive',minimum_term_frequency=5,
    width_in_pixels=1000,term_scorer=ScaledFScorePresetsNeg1To1(beta=1,scaler_algo='normcdf'),
    grey_threshold=0,y_axis_values=[-1,0,1],metadata=all_satisfaction_score_comment_in_all_conds_df['feature'])

# ================================================================================
fn = '/mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext/demo_scaled_f_score.html'
open(fn,'wb').write(html.encode('utf-8'))
print('Open ' + fn + ' in Chrome or Firefox.')