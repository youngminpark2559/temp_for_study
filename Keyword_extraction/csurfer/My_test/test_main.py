# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/Data_mining/Keyword_extraction/csurfer/My_test && \
# rm e.l && python test_main.py \
# 2>&1 | tee -a e.l && code e.l

import sys
network_dir='/mnt/1T-5e7/mycodehtml/Data_mining/Keyword_extraction/csurfer'
sys.path.insert(0,network_dir)

from rake_nltk import Rake

# Uses stopwords for english from NLTK, and all puntuation characters by
# default
r = Rake()

text="""
Symptoms
The symptoms of high blood sugar in type 2 diabetes tend to appear gradually. Not everyone with type 2 diabetes will notice symptoms in the early stages.

If a person does experience symptoms, they may notice the following:

Frequent urination and increased thirst: When excess glucose builds up in the bloodstream, the body will extract fluid from tissues. This can lead to excessive thirst and the need to drink and urinate more.
Increased hunger: In type 2 diabetes, the cells are not able to access glucose for energy. The muscles and organs will be low on energy, and the person may feel more hungry than usual.
Weight loss: When there is too little insulin, the body may start burning fat and muscle for energy. This causes weight loss.
Fatigue: When cells lack glucose, the body becomes tired. Fatigue can interfere with daily life when a person has type 2 diabetes.
Blurred vision: High blood glucose can cause fluid to be pulled from the lenses of the eyes, resulting in swelling, leading to temporarily blurred vision.
Infections and sores: It takes longer to recover from infections and sores because blood circulation is poor and there may be other nutritional deficits.
If people notice these symptoms, they should see a doctor. Diabetes can lead to a number of serious complications. The sooner a person starts to manage their glucose levels, the better chance they have of preventing complications.
"""

# Extraction given the text.
r.extract_keywords_from_text(text)

# Extraction given the list of strings where each string is a sentence.
# r.extract_keywords_from_sentences()

# To get keyword phrases ranked highest to lowest.
print("r.get_ranked_phrases()",r.get_ranked_phrases())
# r.get_ranked_phrases() ['body may start burning fat', 'type 2 diabetes tend', 'type 2 diabetes', 'body becomes tired', 'high blood sugar', 'excess glucose builds', 'high blood glucose', 'temporarily blurred vision', 'causes weight loss', 'person may feel', 'cells lack glucose', 'weight loss', 'blurred vision', 'may notice', 'blood circulation', 'glucose levels', 'access glucose', 'takes longer', 'serious complications', 'preventing complications', 'person starts', 'people notice', 'nutritional deficits', 'little insulin', 'increased thirst', 'increased hunger', 'frequent urination', 'extract fluid', 'excessive thirst', 'early stages', 'daily life', 'cause fluid', 'better chance', 'appear gradually', 'notice symptoms', 'experience symptoms', 'body', 'may', 'diabetes', 'person', 'cells', 'symptoms', 'usual', 'urinate', 'tissues', 'swelling', 'sores', 'sooner', 'see', 'resulting', 'recover', 'pulled', 'poor', 'organs', 'number', 'need', 'muscles', 'muscle', 'manage', 'low', 'lenses', 'leading', 'lead', 'interfere', 'infections', 'hungry', 'following', 'fatigue', 'eyes', 'everyone', 'energy', 'drink', 'doctor', 'bloodstream', 'able']

# To get keyword phrases ranked highest to lowest with scores.
print("r.get_ranked_phrases_with_scores()",r.get_ranked_phrases_with_scores())
# r.get_ranked_phrases_with_scores() [(20.75, 'body may start burning fat'), (13.666666666666666, 'type 2 diabetes tend'), (9.666666666666666, 'type 2 diabetes'), (9.0, 'body becomes tired'), (8.666666666666666, 'high blood sugar'), (8.6, 'excess glucose builds'), (8.266666666666666, 'high blood glucose'), (8.0, 'temporarily blurred vision'), (8.0, 'causes weight loss'), (7.75, 'person may feel'), (7.6, 'cells lack glucose'), (5.0, 'weight loss'), (5.0, 'blurred vision'), (4.75, 'may notice'), (4.666666666666666, 'blood circulation'), (4.6, 'glucose levels'), (4.6, 'access glucose'), (4.0, 'takes longer'), (4.0, 'serious complications'), (4.0, 'preventing complications'), (4.0, 'person starts'), (4.0, 'people notice'), (4.0, 'nutritional deficits'), (4.0, 'little insulin'), (4.0, 'increased thirst'), (4.0, 'increased hunger'), (4.0, 'frequent urination'), (4.0, 'extract fluid'), (4.0, 'excessive thirst'), (4.0, 'early stages'), (4.0, 'daily life'), (4.0, 'cause fluid'), (4.0, 'better chance'), (4.0, 'appear gradually'), (3.666666666666667, 'notice symptoms'), (3.666666666666667, 'experience symptoms'), (3.0, 'body'), (2.75, 'may'), (2.6666666666666665, 'diabetes'), (2.0, 'person'), (2.0, 'cells'), (1.6666666666666667, 'symptoms'), (1.0, 'usual'), (1.0, 'urinate'), (1.0, 'tissues'), (1.0, 'swelling'), (1.0, 'sores'), (1.0, 'sooner'), (1.0, 'see'), (1.0, 'resulting'), (1.0, 'recover'), (1.0, 'pulled'), (1.0, 'poor'), (1.0, 'organs'), (1.0, 'number'), (1.0, 'need'), (1.0, 'muscles'), (1.0, 'muscle'), (1.0, 'manage'), (1.0, 'low'), (1.0, 'lenses'), (1.0, 'leading'), (1.0, 'lead'), (1.0, 'interfere'), (1.0, 'infections'), (1.0, 'hungry'), (1.0, 'following'), (1.0, 'fatigue'), (1.0, 'eyes'), (1.0, 'everyone'), (1.0, 'energy'), (1.0, 'drink'), (1.0, 'doctor'), (1.0, 'bloodstream'), (1.0, 'able')]

