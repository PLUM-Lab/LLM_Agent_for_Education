


###################################
# 3.2.3 Evaluate generated questions
prompt1_evaluate_question_quality = """You will be given a long document, and several questions. Your task is to evaluate the quality of these generated questions based on the document.

Here is the document: $DOCUMENT$

In this task, you need to evaluate the quality of the generated questions based on the document. The quality of the generated questions depends on how meaningful, valuable, and relevant they are to the document.

Provide your rating on a scale from 1 to 5 based on the criteria below:

- **Rating 1**: The question quality is extremely poor. The generated question is completely unrelated to the document.
- **Rating 2**: The question quality is somewhat poor. The generated question is related only to the document or only to the target persona, but not both. The question may also be meaningless in helping the target persona achieve their goals.
- **Rating 3**: The question quality is good. The generated question is related to both the document and the target persona, but the connection is not very strong. The question is somewhat meaningful and can help the persona partially achieve one of their goals. The persona might ask the question, but not always.
- **Rating 4**: The question quality is very good. The generated question is closely related to both the document and the target persona. However, the question is more likely to be asked by "OTHER PERSONAS" than the target persona.
- **Rating 5**: The question quality is excellent. The generated question is highly relevant to both the document and the target persona. The persona will definitely ask the question about the reference document. Compared to "OTHER PERSONAS", the question is more likely to be asked by the target persona.

Here are the generated questions separated by semicolons: $QUESTIONS$.

For each question, conduct the evaluation as described above. If you provide score of 4, also reply which "other persona" is more likely to ask the question compared to the target persona $PERSONA$; if you provide other scores, reply none for this. Your response should be in JSON format, with the question as the key and the score with other persona as the value.

Ensure that the key is an exact copy of the question and the score is between 1 and 5. Ensure the output follows a VALID JSON FORMAT!

Given the example questions: "Question A?; Question B?", the example output is:

```json
{
    "Question A?": [4, "other_persona"],
    "Question B?": [3, "None"]
}
```

The score you give for each question is:
"""

prompt2_evaluate_question_quality = """You will be given a long document, a target persona with specific goals, and several questions that the target persona might ask. Your task is to evaluate the quality of these generated questions based on the document and the target persona's goals.

Here is the document: $DOCUMENT$

Here is the persona: $PERSONA$.

Here are the goals of the target persona: $GOALS$.

Here are the generated questions separated by semicolons: $QUESTIONS$.

In this task, you need to evaluate the quality of the generated questions based on the document and the $PERSONA$'s goals. The quality of the generated questions depends on how meaningful, valuable, and relevant they are to the document and $PERSONA$'s goals.

Provide your rating on a scale from 1 to 5 based on the criteria below:

- **Rating 1**: The question quality is extremely poor. The generated question is completely unrelated to the document and $PERSONA$'s goals.
- **Rating 2**: The question quality is somewhat poor. The generated question is related only to the document or only to $PERSONA$, but not both. The question may also be meaningless in helping $PERSONA$ achieve their goals.
- **Rating 3**: The question quality is good. The generated question is related to both the document and the target persona $PERSONA$, but the connection is not very strong. The question is somewhat meaningful and can help the persona $PERSONA$ partially achieve one of their goals. The persona might ask the question, but not always.
- **Rating 4**: The question quality is very good. The generated question is closely related to both the document and the target persona $PERSONA$. However, compared to the target persona $PERSONA$, the question is more likely to be asked by one of OTHER PERSONAS including $OTHER_PERSONA$.
- **Rating 5**: The question quality is excellent. The generated question is highly relevant to both the document and the target persona $PERSONA$. The persona $PERSONA$ will definitely ask the question about the reference document. Compared to "OTHER PERSONAS" including $OTHER_PERSONA$, the question is more likely to be asked by the target persona $PERSONA$.

For each question, conduct the evaluation as described above. If you provide score of 4, also reply which "other persona" is more likely to ask the question compared to the target persona $PERSONA$; if you provide other scores, reply none for this. Your response should be in JSON format, with the question as the key and the score with other persona as the value.

Ensure that the key is an exact copy of the question and the score is between 1 and 5. Ensure the output follows a VALID JSON FORMAT!

Given the example questions: "Question A?; Question B?", the example output is:

```json
{
    "Question A?": [4, "other_persona"]
    "Question B?": [3, "None"]
}
```

The score you give for each question is:
"""

prompt3_evaluate_question_quality = """You will be given a long document, a target persona with specific goals, and several questions that the target persona might ask. Your task is to evaluate the quality of these generated questions based on the document and the target persona's goals.

Here is the document: $DOCUMENT$

In this task, you need to evaluate the quality of the generated questions based on the document and the persona's goals. The quality of the generated questions depends on how meaningful, valuable, and relevant they are to the document and persona's goals.

Provide your rating on a scale from 1 to 5 based on the criteria below:

- **Rating 1**: The question quality is extremely poor. The generated question is completely unrelated to the document and persona's goals.
- **Rating 2**: The question quality is somewhat poor. The generated question is related only to the document or only to persona, but not both. The question may also be meaningless in helping persona achieve their goals.
- **Rating 3**: The question quality is good. The generated question is related to both the document and the target persona, but the connection is not very strong. The question is somewhat meaningful and can help the persona partially achieve one of their goals. The persona might ask the question, but not always.
- **Rating 4**: The question quality is very good. The generated question is closely related to both the document and the target persona. However, compared to the target persona, the question is more likely to be asked by one of OTHER PERSONAS.
- **Rating 5**: The question quality is excellent. The generated question is highly relevant to both the document and the target persona. The persona will definitely ask the question about the reference document. Compared to "OTHER PERSONAS", the question is more likely to be asked by the target persona.

For each question, conduct the evaluation as described above. If you provide score of 4, also reply which "other persona" is more likely to ask the question compared to the target persona; if you provide other scores, reply none for this. Your response should be in JSON format, with the question as the key and the score with other persona as the value.

Here is the target persona: $PERSONA$.

Here are the goals of the target persona: $GOALS$.

Here are the generated questions separated by semicolons: $QUESTIONS$.

Here are OTHER PERSONAS: $OTHER_PERSONA$.

Ensure that the key is an exact copy of the question and the score is between 1 and 5. Ensure the output follows a VALID JSON FORMAT!

Given the example questions: "Question A?; Question B?", the example output is:

```json
{
    "Question A?": [4, "other_persona"]
    "Question B?": [3, "None"]
}
```

The score you give for each question is:
"""

prompt4_evaluate_question_answerability = """You will be given a long document and several questions related to the document. Your task is to evaluate whether these questions can be answered based on the content of the document. 

Here is the document: $DOCUMENT$

For each question:

1. If the document contains the answer, provide the answer and the exact reference text from the document. The answer should not be a direct copy from the original document. You should answer the question in your own words but refer to the document contents. The reference text should contain enough information to answer the question. If the reference texts contain different parts, concatenate every parts together.
2. If the document does not contain the answer, return "None" for both the answer and the reference.

You will be given several questions to evaluate. Conduct the task described above for each question. Your response should be in JSON format, with each question as the key and the answer and reference as the values.

Ensure that the key is an exact copy of the question and the reference is an exact copy of a text span in the given document. Ensure the output follows a VALID JSON FORMAT!

Example of two questions (the first question is answerable, while the second one is not answerable):

**Questions:**

1. Question 1?
2. Question 2?

**Answers:**
```json
{
    "Question 1?": { "Answer": "xxx", "Reference": "yyy" },
    "Question 2?": { "Answer": "None", "Reference": "None" },
}
```

**Questions:**
$QUESTIONS$

**Answers:**
"""

prompt5_evaluate_question_quality_choose_persona = """You will be given a summary of a document, one question and several personas. Your task is to conduct a multiple choice to choose the personas that might be interested in the given question that is related to the document. You should respond in a JSON format.

Here is an example. In  this example, four personas are given to you, and the persona3 is the most one to be interested in the question, while the persona2 is the second one. Persona1 and persona4 are not interested in the question. Example of the INPUT and OUTPUT:

**INPUT**:

**Document**:

Document content.

**Question**:

Question?

**Personas**

Persona1, persona2, persona3, persona4.

**OUTPUT**:
```json
{
    "order 1": "persona3,
    "order 2": "persona2"
}
```
**INPUT**:

**Document**:

$DOCUMENT$

**Question**:

$QUESTION$

**Personas**

$PERSONA$

**OUTPUT**:
"""  # also give some very general questions, e.g., list 5 key points, give me a summary; give one invalid question, totally not related to every persona


prompts_template_dict = {
    "prompt1_evaluate_question_quality": prompt1_evaluate_question_quality,
    "prompt2_evaluate_question_quality": prompt2_evaluate_question_quality,
    "prompt3_evaluate_question_quality": prompt3_evaluate_question_quality,
    "prompt4_evaluate_question_answerability": prompt4_evaluate_question_answerability,
    "prompt5_evaluate_question_quality_choose_persona": prompt5_evaluate_question_quality_choose_persona,

}
