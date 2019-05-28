# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext && \
# rm e.l && python demo_phrase_machine.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import spacy

from scattertext import SampleCorpora, PhraseMachinePhrases
from scattertext import produce_scattertext_explorer
from scattertext.CorpusFromPandas import CorpusFromPandas
from scattertext.termcompaction.CompactTerms import CompactTerms

import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns',None)
from Data.utils import utils_data as utils_data

# ================================================================================
# convention_df = SampleCorpora.ConventionData2012.get_data()
# corpus = (CorpusFromPandas(convention_df,
#                            category_col='party',
#                            text_col='text',
#                            feats_from_spacy_doc=PhraseMachinePhrases(),
#                            nlp=spacy.load('en', parser=False))
#           .build()
#           .compact(CompactTerms(minimum_term_count=2)))

# html = produce_scattertext_explorer(corpus,
#                                     category='democrat',
#                                     category_name='Democratic',
#                                     not_category_name='Republican',
#                                     minimum_term_frequency=2,
#                                     pmi_threshold_coefficient=0,
#                                     width_in_pixels=1000,
#                                     metadata=convention_df['speaker'])
# open('./demo_phrase_machine.html', 'wb').write(html.encode('utf-8'))
# print('Open ./demo_phrase_machine.html in Chrome or Firefox.')

# ================================================================================
all_satisfaction_score_comment_in_all_conds=utils_data.get_all_satisfaction_score_comment_in_all_conds()

columns=['senti_on_Metfor_oral','feature','review']
all_satisfaction_score_comment_in_all_conds_df=pd.DataFrame(all_satisfaction_score_comment_in_all_conds,index=None,columns=columns)

# ================================================================================
corpus=(CorpusFromPandas(
  all_satisfaction_score_comment_in_all_conds_df,category_col='senti_on_Metfor_oral',text_col='review',
  feats_from_spacy_doc=PhraseMachinePhrases(),nlp=spacy.load('en',parser=False)).build().compact(CompactTerms(minimum_term_count=2)))

# ================================================================================
html=produce_scattertext_explorer(
  corpus,category='negative',category_name='Negative',not_category_name='Positive',minimum_term_frequency=2,
  pmi_threshold_coefficient=0,width_in_pixels=1000,metadata=all_satisfaction_score_comment_in_all_conds_df['feature'])

# ================================================================================
open('/mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext/demo_phrase_machine.html','wb').write(html.encode('utf-8'))
print('Open ./demo_phrase_machine.html in Chrome or Firefox.')
