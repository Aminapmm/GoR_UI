QUERY_GENERATE = """
You are a great questioner of any text, and are adept at asking valuable and insightful questions. 
Your goal is to generate 1 summary question for the text provided below. 
The generated summary question should try to simulate the tone of human questions as much as possible, 
and make sure that the generated question must be interrogative sentences and a summary question. 
Important! Please make sure this text must be a complete and non-redundant answer to the generated summary question. 
Please directly output the generated summary question, do not output irrelevant text.

DOCUMENT:
{document}
"""


QUERY_PROMPT_QMSUM = """
Refer to the following meeting transcripts and answer the question with brief but complete explanations. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_QMSUM_NORMAL = """
Refer to the following meeting transcripts and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_SQuALITY = """
Refer to the following story and answer the question with brief but complete explanations. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_SQuALITY_NORMAL = """
Refer to the following story and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_GOV = """
Refer to the following report and answer the question with brief but complete explanations. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_GOV_NORMAL = """
Refer to the following report and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_WCEP = """
Refer to the following document and answer the question with brief but complete explanations. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_WCEP_NORMAL = """
Refer to the following document and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_BOOK = """
Refer to the following narrative and answer the question with brief but complete explanations. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_BOOK_NORMAL = """
Refer to the following narrative and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT = {
    "qmsum": QUERY_PROMPT_QMSUM,
    "wcep": QUERY_PROMPT_WCEP,
    "booksum": QUERY_PROMPT_BOOK,
    "govreport": QUERY_PROMPT_GOV,
    "squality": QUERY_PROMPT_SQuALITY
}


QUERY_PROMPT_NORMAL = {
    "qmsum": QUERY_PROMPT_QMSUM_NORMAL,
    "wcep": QUERY_PROMPT_WCEP_NORMAL,
    "booksum": QUERY_PROMPT_BOOK_NORMAL,
    "govreport": QUERY_PROMPT_GOV_NORMAL,
    "squality": QUERY_PROMPT_SQuALITY_NORMAL
}


if __name__ == '__main__':
    pass
