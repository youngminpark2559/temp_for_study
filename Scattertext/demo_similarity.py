# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext && \
# rm e.l && python demo_similarity.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import spacy

from Data.utils import utils_data as utils_data
from scattertext import SampleCorpora, word_similarity_explorer
from scattertext.CorpusFromPandas import CorpusFromPandas
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns',None)

# ================================================================================
def main():
  nlp = spacy.load('en')

  # ================================================================================
  # convention_df = SampleCorpora.ConventionData2012.get_data()
  # print("convention_df",convention_df)
  #         party         speaker  \
  # 0    democrat    BARACK OBAMA               
  # 1    democrat    MICHELLE OBAMA             
  # 2    democrat    RICHARD DURBIN             
  # 3    democrat    JOSEPH BIDEN               
  # 4    democrat    JILL BIDEN                 
  # 5    democrat    ANGIE FLORES               
  
  #      text
  # 0    Thank you. Thank you. Thank you. Thank you so much.Thank you.Thank you so much. Thank you. Thank you very 
  # 1    Thank you so much. Tonight, I am so thrilled and so honored and so proud to introduce the love of my life 
  # 2    Thank you. It is a singular honor to be here tonight. Eight years ago in Boston, I introduced you to a sta
  # 3    Hey, Delaware. \nAnd my favorite Democrat, Jilly, I want you to know that Beau and Hunt and Ashley and I —
  # 4    Hello. \nThank you, Angie. I'm so proud of how far you've come.\nI'm so proud to stand before you tonight 
  # 5    My name is Angie Flores and I am a student at Miami-Dade College. \nWhen you grow up in a family where get

  # print("convention_df",convention_df.shape)
  # (189, 3)

  # df1=convention_df.iloc[:10,:]
  # df2=convention_df.iloc[150:160,:]
  # df_cat=pd.concat([df1,df2],axis=0)
  # # print("df_cat",df_cat.shape)
  # # (20, 3)
  # convention_df=df_cat

  # ================================================================================
  # convention_df=pd.read_csv('/mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext/Data/WebMD_Metformin_oral/text.csv',encoding='utf8',error_bad_lines=False)
  # print("convention_df",convention_df.shape)

  # ================================================================================
  # corpus = CorpusFromPandas(convention_df,
  #                           category_col='party',
  #                           text_col='text',
  #                           nlp=nlp).build()
  
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
  corpus=CorpusFromPandas(
    all_satisfaction_score_comment_in_all_conds_df,category_col='senti_on_Metfor_oral',text_col='review',nlp=nlp).build()
  
  # ================================================================================
  html=word_similarity_explorer(
    corpus,category='negative',category_name='Negative',not_category_name='Positive',target_term='jobs',
    minimum_term_frequency=5,width_in_pixels=1000,metadata=all_satisfaction_score_comment_in_all_conds_df['feature'],
    alpha=0.01,max_p_val=0.1,save_svg_button=True)
  
  # ================================================================================
  open('/mnt/1T-5e7/mycodehtml/Data_mining/Visualization/Scattertext/demo_similarity.html', 'wb').write(html.encode('utf-8'))
  print('Open ./demo_similarlity.html in Chrome or Firefox.')

# ================================================================================
if __name__ == '__main__':
  main()
