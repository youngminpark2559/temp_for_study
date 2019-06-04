#!/usr/bin/env python
# coding:utf-8

# export JAVAHOME=/home/young/Downloads/Fiji.app/java/linux-amd64/jdk1.8.0_172/jre/bin

# conda activate py27_keyword_extraction && \
# cd /mnt/1T-5e7/mycodehtml/Data_mining/Keyword_extraction/lvsh && \
# rm e.l && python example.py /mnt/1T-5e7/mycodehtml/Data_mining/Keyword_extraction/lvsh/data/type2_diabetes.txt \
# 2>&1 | tee -a e.l && code e.l

"""
Runs the keyword extraction algorithmn on an example
"""
__author__ = "Lavanya Sharan"


import sys
from keywordextraction import *

import nltk; nltk.download('all')

def main():
  if len(sys.argv)==1:
    raise ValueError('Must specify input text file.')  
  else:
    f = open(sys.argv[1],'r')
    text = f.read()
    f.close()

  # load keyword classifier
  preload = 1
  classifier_type = 'logistic'
  keyword_classifier = get_keywordclassifier(preload,classifier_type)['model']

  # extract top k keywords
  top_k = 15
  keywords = extract_keywords(text,keyword_classifier,top_k,preload)  
  print 'ORIGINAL TEXT:\n%s\nTOP-%d KEYWORDS returned by model: %s\n' % (text,top_k,', '.join(keywords))
  
  # ================================================================================
  # ORIGINAL TEXT:
  # What Causes Diabetes?
  # Your pancreas makes a hormone called insulin. It's what lets your cells turn glucose from the food you eat into energy. People with type 2 diabetes make insulin, but their cells don't use it as well as they should. Doctors call this insulin resistance.

  # At first, the pancreas makes more insulin to try to get glucose into the cells. But eventually it can't keep up, and the sugar builds up in your blood instead.

  # Usually a combination of things cause type 2 diabetes, including:

  # Genes. Scientists have found different bits of DNA that affect how your body makes insulin.

  # Extra weight. Being overweight or obese can cause insulin resistance, especially if you carry your extra pounds around the middle. Now type 2 diabetes affects kids and teens as well as adults, mainly because of childhood obesity.

  # Metabolic syndrome. People with insulin resistance often have a group of conditions including high blood glucose, extra fat around the waist, high blood pressure, and high cholesterol and triglycerides.

  # Too much glucose from your liver. When your blood sugar is low, your liver makes and sends out glucose. After you eat, your blood sugar goes up, and usually the liver will slow down and store its glucose for later. But some people's livers don't. They keep cranking out sugar.

  # Bad communication between cells. Sometimes cells send the wrong signals or don't pick up messages correctly. When these problems affect how your cells make and use insulin or glucose, a chain reaction can lead to diabetes.

  # Broken beta cells. If the cells that make the insulin send out the wrong amount of insulin at the wrong time, your blood sugar gets thrown off. High blood glucose can damage these cells, too.


  # TOP-15 KEYWORDS returned by model: 
  # insulin, communication, resistance, combination, diabetes, scientists, cholesterol, pancreas, cells, metabolic, conditions, different, childhood, doctors, hormone

  # ================================================================================
  # evaluate performance for inspec example
  if sys.argv[1]=='inspec.txt':
    true_keywords = []
    with open('inspec.key','r') as f:
      for line in f:
        true_keywords.extend(line.strip('\n').split())
    true_keywords = [remove_punctuation(kw.lower()) for kw in true_keywords]

    (precision,recall,f1score) = evaluate_keywords(keywords,true_keywords)
    print 'MANUALLY SPECIFIED KEYWORDS:\n%s' % ', '.join(true_keywords)
    print '\nModel achieves %.4f precision, %.4f recall, and %.4f f1 score.' % (precision,recall,f1score)


if __name__ == '__main__':
  main()
