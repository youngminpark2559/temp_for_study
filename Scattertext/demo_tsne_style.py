# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext && \
# rm e.l && python demo_tsne_style.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import scattertext as st
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns',None)
from Data.utils import utils_data as utils_data

# pip install umap-learn

# ================================================================================
# convention_df = st.SampleCorpora.ConventionData2012.get_data()
# convention_df['parse'] = convention_df['text'].apply(st.whitespace_nlp_with_sentences)

# corpus = (st.CorpusFromParsedDocuments(convention_df,
#                                        category_col='party',
#                                        parsed_col='parse')
#           .build().get_stoplisted_unigram_corpus())


# html = st.produce_projection_explorer(corpus,
#                                       category='democrat',
#                                       category_name='Democratic',
#                                       not_category_name='Republican',
#                                       metadata=convention_df.speaker,
#                                       width_in_pixels=1000)

# ================================================================================
all_satisfaction_score_comment_in_all_conds=utils_data.get_all_satisfaction_score_comment_in_all_conds()

# ================================================================================
columns=['senti_on_Metfor_oral','feature','review']
all_satisfaction_score_comment_in_all_conds_df=pd.DataFrame(all_satisfaction_score_comment_in_all_conds,index=None,columns=columns)

all_satisfaction_score_comment_in_all_conds_df['parse'] = all_satisfaction_score_comment_in_all_conds_df['review'].apply(st.whitespace_nlp_with_sentences)

# ================================================================================
corpus=(st.CorpusFromParsedDocuments(
  all_satisfaction_score_comment_in_all_conds_df,category_col='senti_on_Metfor_oral',parsed_col='parse').build().get_stoplisted_unigram_corpus())

# ================================================================================
html=st.produce_projection_explorer(
  corpus,category='negative',category_name='Negative',not_category_name='Positive',
  metadata=all_satisfaction_score_comment_in_all_conds_df.feature,width_in_pixels=1000)

# ================================================================================
file_name='/mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext/demo_tsne_style.html'
open(file_name,'wb').write(html.encode('utf-8'))
print('Open',file_name,'in chrome')

