# coding: utf-8
#!/usr/bin/env python
import numpy as np
import os
import openai
import requests
from sentence_transformers import SentenceTransformer, util
import streamlit as st


openai.api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(layout="wide")


def andize(lst):
    if len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return lst[0] + " and " + lst[1]
    elif len(lst) <= 5:
        return ', '.join(lst[:-1]) + " and " + lst[-1]
    else:
        return ', '.join(lst[:4]) + f" and {len(lst) - 4} others"

@st.cache
def get_papers(author_id):
    endpoint = f"https://api.semanticscholar.org/graph/v1/author/{author_id}?fields=name,papers.title,papers.abstract,papers.authors"
    data = requests.get(endpoint).json()
    return data['papers'], data['name']


def get_top_n(dot_scores, n=3):
    # Use a crossing-out method.
    n = min(n, min(dot_scores.shape))
    matches = []
    for i in range(n):
        the_max = np.max(dot_scores)
        ci, cj = np.where(dot_scores == the_max)
        
        dot_scores[ci, :] = 0
        dot_scores[:, cj] = 0
        
        matches.append((ci[0], cj[0], the_max))
        
    return matches

def zero_out(codes, papers):
    for i, p in enumerate(papers):
        if p['abstract'] is None:
            codes[i, :] = 0
    return codes


# Extract papers in common
def get_coauthors(papers):
    authors = []
    for p in papers:
        authors += p['authors']
    return {x['authorId']: x['name'] for x in authors}


def report_common_papers(p1, p2):
    p1s = set([x['paperId'] for x in p1])
    p2s = set([x['paperId'] for x in p2])
    
    common_papers = p1s.intersection(p2s)
    if len(common_papers) > 0:
        common_paper_titles = ['"' + x['title'] + '"' for x in p1 if x['paperId'] in common_papers]
        titles = np.unique(common_paper_titles)
        the_str = "💡 You co-wrote " + andize(titles)
    else:
        the_str = ""
    return the_str, common_papers

def report_common_coauthors(p1, p2):
    coa_1 = get_coauthors(p1)
    coa_2 = get_coauthors(p2)
    
    coa_1_ids = set(coa_1.keys())
    coa_2_ids = set(coa_2.keys())
    
    common_coauthors = coa_1_ids.intersection(coa_2_ids)
    if None in common_coauthors:
        common_coauthors = common_coauthors - {None}
        
    print(common_coauthors)
    
    if len(common_coauthors) > 0:
        names = [coa_1[x] for x in common_coauthors]
        the_str = "🎉 You co-wrote papers with " + andize(names)
    else:
        the_str = ""
    return the_str

def remove_common_papers(papers, bad_ids):
    papers_out = []
    for p in papers:
        if p['paperId'] in bad_ids:
            continue
        papers_out.append(p)
    return papers_out

def remove_noabstract_papers(papers):
    papers_out = []
    for p in papers:
        if p['abstract'] is None:
            continue
        papers_out.append(p)
    return papers_out


def encode_papers(papers):
    lst = []
    for p in papers:
        if p['abstract'] is None:
            lst.append(p['title'])
        else:
            lst.append(p['title'] + '\n' + p['abstract'])
    return lst

@st.cache
def report_common_topics(t1, t2):
    # Use specter to embed the papers that we care about
    model = SentenceTransformer('allenai-specter')
    
    # Transform all the papers that you have in common.
    papers_1_encoded = encode_papers(t1)
    papers_2_encoded = encode_papers(t2)
    
    codes_1 = model.encode(papers_1_encoded)
    codes_2 = model.encode(papers_2_encoded)
    
    return codes_1, codes_2

def make_single_prompt(abstract):
    return "Read this scientific abstract and describe 3 salient points from it.\n\nAbstract:" + abstract + "\n\nThis abstract deals with:\n-"


def make_dual_prompt(bp0, bp1):
    text = "Abstract 1 deals with:\n-" + "\n-".join(bp0)
    text += "\n\nAbstract 2 deals with:\n-" + "\n-".join(bp1)
    text += "\n\nAbstract 1 and abstract 2 both deal with the following broad topics:\n-"
    return text


def clean_bullet_points(bp):
    bp = ("-" + bp).split('\n')
    bps = []
    for x in bp:
        x = x.strip()
        if not x:
            continue
            
        if not x.startswith('-'):
            continue
            
        x = x[1:].strip()
            
        if x.endswith('.') or x.endswith(','):
            x = x[:-1]
        x = x[0].upper() + x[1:]
        bps.append(x)
    return bps

@st.cache
def summarize_one_abstract(text):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=make_single_prompt(text),
      temperature=0.2,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response['choices'][0]['text']

def summarize_two_abstracts(bp0, bp1):
    prompt = make_dual_prompt(bp0, bp1)
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=prompt,
      temperature=0.2,
      max_tokens=48,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response['choices'][0]['text']

def main():
    col0, col1 = st.columns(2)

    database = {
        "Patrick Mineault": "2147497",
        "Kate Bonnen": "2355919",
        "Dan Goodman": "1866844",
        "Megan Peters": "47750194",
        "Konrad Kording": "3282030",
        "Anne Urai": "3174285",
        "Brad Wyble": "46520750",
        "Chris Rozell": "1690427",
        "Titipat Achakulvisat": "3393424",
        "Ioana Marinescu": "4386407",
        "Custom person (paste semantic scholar id)": ""
    }

    people = database.keys()

    sel0 = col0.selectbox("Person 1", people)
    sel1 = col1.selectbox("Person 2", people, 1)

    id0 = col0.text_input("Semantic scholar id 1", database[sel0])
    id1 = col1.text_input("Semantic scholar id 2", database[sel1])

    if id0 == "" or id1 == "":
        return

    papers_0, author_0 = get_papers(id0)
    papers_1, author_1 = get_papers(id1)

    col0.write(author_0)
    col1.write(author_1)

    col0.write(f"Found {len(papers_0)} papers on Semantic scholar")
    col1.write(f"Found {len(papers_1)} papers on Semantic scholar")

    c = st.container()

    c.write("## Here's what you have in common")

    common_paper_str, common_papers = report_common_papers(papers_0, papers_1)
    papers_clean_0 = remove_common_papers(papers_0, common_papers)
    papers_clean_1 = remove_common_papers(papers_1, common_papers)

    c.write(common_paper_str)
    c.write(report_common_coauthors(papers_clean_0, papers_clean_1))

    papers_clean_0 = remove_noabstract_papers(papers_clean_0)
    papers_clean_1 = remove_noabstract_papers(papers_clean_1)

    codes_0, codes_1 = report_common_topics(papers_clean_0, papers_clean_1)

    c.write("## Your most similar papers")

    dot_scores = util.dot_score(codes_0, codes_1).cpu().numpy()

    best_matches = get_top_n(dot_scores)
    
    for i, m in enumerate(best_matches):
        c.write(f"Match score: {m[2]:.1f}")
        col0, col1 = st.columns(2)

        col0.write("### " + papers_clean_0[m[0]]['title'] )
        col1.write("### " + papers_clean_1[m[1]]['title'] )

        summary_0 = summarize_one_abstract(papers_clean_0[m[0]]['abstract'])
        summary_1 = summarize_one_abstract(papers_clean_1[m[1]]['abstract'])

        bullet_points_0 = clean_bullet_points(summary_0)
        bullet_points_1 = clean_bullet_points(summary_1)

        joint_summary = summarize_two_abstracts(bullet_points_0, bullet_points_1)
        joint_bullet_point = clean_bullet_points(joint_summary)

        col0.write("It's about: ")
        col0.write("- " + "\n- ".join(bullet_points_0))

        col1.write("It's about: ")
        col1.write("- " + "\n- ".join(bullet_points_1))

        c = st.container()
        c.write("They're both about: ")
        c.write("- " + "\n- ".join(joint_bullet_point))

        c.write("---")

main()