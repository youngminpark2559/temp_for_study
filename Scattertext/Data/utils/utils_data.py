
# ================================================================================
import pickle

# ================================================================================
from Data.utils import utils_common as utils_common

# ================================================================================
Condition_type_2_diabetes_mellitus="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_type_2_diabetes_mellitus"
Condition_prevention_of_type2_diabetes_mellitus="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_prevention_of_type2_diabetes_mellitus"
Condition_other="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_other"
Condition_disease_of_ovaries_with_cysts="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_disease_of_ovaries_with_cysts"
Condition_diabetes_during_pregnancy="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_diabetes_during_pregnancy"

comments_data=[
  Condition_type_2_diabetes_mellitus,
  Condition_prevention_of_type2_diabetes_mellitus,
  Condition_other,
  Condition_disease_of_ovaries_with_cysts]

# ================================================================================
def get_overall_stat(one_condition_entire_data):
  overall_stat=one_condition_entire_data.pop(0)
  # print("overall_stat",overall_stat)
  # ['Effectiveness@@@@@Current Rating: 0@@@@@(3.29)', 'Ease of Use@@@@@Current Rating: 0@@@@@(3.91)', 'Satisfaction@@@@@Current Rating: 0@@@@@(2.90)']

  overall_stat_refined=[]
  for one_score in overall_stat:
    replaced=one_score.replace("Effectiveness@@@@@Current Rating: 0@@@@@(","").replace("Ease of Use@@@@@Current Rating: 0@@@@@(","").replace("Satisfaction@@@@@Current Rating: 0@@@@@(","").replace(")","")
    # print("replaced",replaced)
    # 3.29

    overall_stat_refined.append(replaced)
  
  return overall_stat_refined

def get_satisfaction_score_and_comment(one_condition_entire_data):
  all_satisfaction_score_comment_in_all_page=[]
  for all_reviews_in_one_page in one_condition_entire_data:
    # print("all_reviews_in_one_page",all_reviews_in_one_page)
    # ['@@@@@Page0', ['@@@@@Post0', 'Condition: Type 2 Diabetes Mellitus', '5/24/2019 5:54:59 PM', 'Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)', 'Effectiveness@@@@@1', 'Ease 

    page_num=all_reviews_in_one_page.pop(0)
    all_reviews_in_one_page_wo_page_num=all_reviews_in_one_page
    # print("all_reviews_in_one_page_wo_page_num",all_reviews_in_one_page_wo_page_num)
    # [['@@@@@Post0', 'Condition: Type 2 Diabetes Mellitus', '5/24/2019 5:54:59 PM', 'Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)', 'Effectiveness@@@@@1', 'Ease of Use@@@@@5', 

    # ================================================================================
    all_satisfaction_score_comment_in_one_page=[]
    for one_review_in_one_page in all_reviews_in_one_page_wo_page_num:
      # print("one_review_in_one_page",one_review_in_one_page)
      # ['@@@@@Post0', 'Condition: Type 2 Diabetes Mellitus', '5/24/2019 5:54:59 PM', 'Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)', 'Effectiveness@@@@@1', 'Ease of Use@@@@@5', 

      satisfaction_score=int(one_review_in_one_page[6].replace("Satisfaction@@@@@",""))
      if satisfaction_score>3:
        satisfaction_score="positive"
      else:
        satisfaction_score="negative"
      # print("satisfaction_score",satisfaction_score)
      # negative

      # ================================================================================
      feature_name=one_review_in_one_page[6].split("@@@@@")[0]
      # print("feature_name",feature_name)
      # Satisfaction

      # ================================================================================
      comment_txt=one_review_in_one_page[7].replace("Comment:","")
      # print("comment_txt",comment_txt)
      # after a week----mouth ulccers,cudnt talk,eat,drink for 5 days....whole body burnt,headache, fatigue....quit---am slowly getting better, wudnt give to my worst enemy.

      if comment_txt=="":
        continue

      # ================================================================================
      all_satisfaction_score_comment_in_one_page.append([satisfaction_score,feature_name,comment_txt])

    all_satisfaction_score_comment_in_all_page.extend(all_satisfaction_score_comment_in_one_page)
  
  # print("all_satisfaction_score_comment_in_all_page",all_satisfaction_score_comment_in_all_page)
  # [['negative', 'after a week----mouth ulccers,cudnt talk,eat,drink for 5 days....whole body burnt,headache, fatigue....quit---am slowly getting better, wudnt give to my worst enemy.'], 

  return all_satisfaction_score_comment_in_all_page

def get_all_satisfaction_score_comment_in_all_conds():
  # print("comments_data",comments_data)
  # ['/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_type_2_diabetes_mellitus', 
  #  '/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_prevention_of_type2_diabetes_mellitus', 
  #  '/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_other', 
  #  '/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_disease_of_ovaries_with_cysts']

  all_satisfaction_score_comment_in_all_conds=[]

  for one_cond in comments_data:

    # print("one_cond",one_cond)
    # /mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_type_2_diabetes_mellitus

    cond_name=one_cond.split("/")[-1]

    # ================================================================================
    loaded_path=utils_common.get_file_list(one_cond+"/*.pkl")
    # print("loaded_path",loaded_path)
    # ['/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_type_2_diabetes_mellitus/comment_texts_entire_stat.pkl', 
    #  '/mnt/1T-5e7/Companies/

    one_condition_entire_data=[]
    for one_path in loaded_path:
      with open(one_path,'rb') as f:
        mynewlist=pickle.load(f)
        # print("mynewlist",mynewlist)
        # ['Effectiveness@@@@@Current Rating: 0@@@@@(3.29)', 'Ease of Use@@@@@Current Rating: 0@@@@@(3.91)', 'Satisfaction@@@@@Current Rating: 0@@@@@(2.90)']

        one_condition_entire_data.append(mynewlist)
    # print("one_condition_entire_data",one_condition_entire_data)
    # [['Effectiveness@@@@@Current Rating: 0@@@@@(3.29)', 'Ease of Use@@@@@Current Rating: 0@@@@@(3.91)', 'Satisfaction@@@@@Current Rating: 0@@@@@(2.90)'], ['@@@@@Page0', ['@@@@@Post0', 'Condition: 

    # ================================================================================
    overall_stat_refined=get_overall_stat(one_condition_entire_data)
    # print("overall_stat_refined",overall_stat_refined)
    # ['3.29', '3.91', '2.90']

    # ================================================================================
    all_satisfaction_score_comment_in_all_page=get_satisfaction_score_and_comment(one_condition_entire_data)
    # print("all_satisfaction_score_comment_in_all_page",len(all_satisfaction_score_comment_in_all_page))
    # Condition_type_2_diabetes_mellitus 990
    # Condition_prevention_of_type2_diabetes_mellitus 107
    # Condition_other 127
    # Condition_disease_of_ovaries_with_cysts 178
  
    # ================================================================================
    all_satisfaction_score_comment_in_all_conds.extend(all_satisfaction_score_comment_in_all_page)

  # ================================================================================
  # print("all_satisfaction_score_comment_in_all_conds",all_satisfaction_score_comment_in_all_conds)
  # print("len(all_satisfaction_score_comment_in_all_conds)",len(all_satisfaction_score_comment_in_all_conds))
  # 1402

  return all_satisfaction_score_comment_in_all_conds


    
    


        




    

