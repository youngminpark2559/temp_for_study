from __future__ import print_function

from scattertext import CorpusFromParsedDocuments, produce_scattertext_explorer
from scattertext import FeatsFromOnlyEmpath
from scattertext import SampleCorpora
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns',None)

from Data.utils import utils_data as utils_data

# ================================================================================
# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext && \
# rm e.l && python demo_empath.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
def main():
  # convention_df = SampleCorpora.ConventionData2012.get_data()
  feat_builder = FeatsFromOnlyEmpath()
  # corpus = CorpusFromParsedDocuments(convention_df,
  #                                    category_col='party',
  #                                    parsed_col='text',
  #                                    feats_from_spacy_doc=feat_builder).build()
  # html = produce_scattertext_explorer(corpus,
  #                                     category='democrat',
  #                                     category_name='Democratic',
  #                                     not_category_name='Republican',
  #                                     width_in_pixels=1000,
  #                                     metadata=convention_df['speaker'],
  #                                     use_non_text_features=True,
  #                                     use_full_doc=True,
  #                                     topic_model_term_lists=feat_builder.get_top_model_term_lists())

  # ================================================================================
  all_satisfaction_score_comment_in_all_conds=utils_data.get_all_satisfaction_score_comment_in_all_conds()
  # print("all_satisfaction_score_comment_in_all_conds",all_satisfaction_score_comment_in_all_conds)
  # [['negative', 'Satisfaction', 'after a week----mouth ulccers,cudnt talk,eat,drink for 5 days....whole body burnt,headache, fatigue....quit---am slowly getting better, wudnt give to my worst 

  # print("all_satisfaction_score_comment_in_all_conds",len(all_satisfaction_score_comment_in_all_conds))
  # 1402
  
  # ================================================================================
  columns=['senti_on_Metfor_oral','feature','review']
  all_satisfaction_score_comment_in_all_conds_df=pd.DataFrame(all_satisfaction_score_comment_in_all_conds,index=None,columns=columns)

  # ================================================================================
  corpus=CorpusFromParsedDocuments(
		all_satisfaction_score_comment_in_all_conds_df,category_col='senti_on_Metfor_oral',
		parsed_col='review',feats_from_spacy_doc=feat_builder).build()

  # ================================================================================
  html=produce_scattertext_explorer(
		corpus,category='negative',category_name='Negative',not_category_name='Positive',width_in_pixels=1000,
		metadata=all_satisfaction_score_comment_in_all_conds_df['feature'],use_non_text_features=True,
		use_full_doc=True,topic_model_term_lists=feat_builder.get_top_model_term_lists())

  # ================================================================================
  open('/mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext/Convention-Visualization-Empath.html', 'wb').write(html.encode('utf-8'))
  print('Open ./Convention-Visualization-Empath.html in Chrome or Firefox.')

# ================================================================================
if __name__ == '__main__':
  main()
