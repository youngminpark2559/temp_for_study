# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/Data_mining/Keyword_extraction/davidadamojr && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import click
import textrank
from textrank import extract_key_phrases

# ================================================================================
@click.group()
def cli():
    pass


@cli.command()
def initialize():
    """Download required nltk libraries."""
    textrank.setup_environment()


@cli.command()
@click.argument('filename')
def extract_summary(filename):
    """Print summary text to stdout."""
    with open(filename) as f:
        summary = textrank.extract_sentences(f.read())
        print(summary)


@cli.command()
@click.argument('filename')
def extract_phrases(filename):
    """Print key-phrases to stdout."""
    with open(filename) as f:
        phrases = textrank.extract_key_phrases(f.read())
        print(phrases)


your_text="""
What Causes Diabetes?
Your pancreas makes a hormone called insulin. It's what lets your cells turn glucose from the food you eat into energy. People with type 2 diabetes make insulin, but their cells don't use it as well as they should. Doctors call this insulin resistance.

At first, the pancreas makes more insulin to try to get glucose into the cells. But eventually it can't keep up, and the sugar builds up in your blood instead.

Usually a combination of things cause type 2 diabetes, including:
Genes. Scientists have found different bits of DNA that affect how your body makes insulin.

Extra weight. Being overweight or obese can cause insulin resistance, especially if you carry your extra pounds around the middle. Now type 2 diabetes affects kids and teens as well as adults, mainly because of childhood obesity.
Metabolic syndrome. People with insulin resistance often have a group of conditions including high blood glucose, extra fat around the waist, high blood pressure, and high cholesterol and triglycerides.

Too much glucose from your liver. When your blood sugar is low, your liver makes and sends out glucose. After you eat, your blood sugar goes up, and usually the liver will slow down and store its glucose for later. But some people's livers don't. They keep cranking out sugar.

Bad communication between cells. Sometimes cells send the wrong signals or don't pick up messages correctly. When these problems affect how your cells make and use insulin or glucose, a chain reaction can lead to diabetes.

Broken beta cells. If the cells that make the insulin send out the wrong amount of insulin at the wrong time, your blood sugar gets thrown off. High blood glucose can damage these cells, too.
"""

keywords_dict=extract_key_phrases(your_text)
# print("keywords_dict",keywords_dict)
# {'Diabetes', 'communication', 'hormone', 'middle', 'glucose', 'Metabolic syndrome', 'combination', 'Broken', 'cholesterol', 'group', 'liver', 'extra', 'different', 'pressure', 'Extra weight', 'insulin resistance', 'insulin', 'sugar', 'pancreas', 'DNA', 'amount', 'energy', 'overweight', 'childhood obesity', 'reaction'}

