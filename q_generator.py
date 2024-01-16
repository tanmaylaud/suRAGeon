import google.generativeai as genai
import time
import pandas as pd
import json
from tqdm import tqdm
# Replace with your actual API key and ensure it is kept confidential
GOOGLE_API_KEY = '<google api key>'
genai.configure(api_key=GOOGLE_API_KEY)
df = pd.read_json('./documents.jsonl', lines=True)
# Define the query function with a delay
def query_gemini(input_paragraph):
    model = genai.GenerativeModel('gemini-pro')

    # Construct the prompt with the new instructions
    prompt = ("Your job is to create training data for a smaller LLM that "
              "will be fine-tuned on the knowledge of a medical textbook. You will be"
              "given a paragraph of a medical textbook called the Open Access "
              "Atlas of Otolaryngology, Head & Neck Operative Surgery and tasked"
              "with creating a question and answers geared at a medical professional"
              "(i.e. general physician/surgeon from a third-world country who may" 
              "not be specialized in otolaryngology, head, and neck operative surgery"
              "For each paragraph, generate a question aimed at probing the knowledge of" 
              "the above described professional and four possible answers, with only one" 
              "them being correct. Then provide the correct answer number and an explanation"
              "of why this is the correct answer \n"
              "It is essential that you put each segment of your response (question, answers, correct answer number, and explanation) on newlines as demonstrated below"
              "Further it is essential to follow the response type with a `:`."
              "Lastly, in your response, do not explicitly mention that the information is from a paragraph from the book or in your question mention the textbook. Lastly, try and keep your questions to be medically relevant."
              "The input will appear as Input Paragraph: ... \n"
              "Your output should be presented in the following way: \n"
              "Question: Your generated question \n"
              "Answer 1: Your first generated answer to the question\n"              
              "Answer 2: Your second generated answer to the question\n"
              "Answer 3: Your third generated answer to the question\n"
              "Answer 4: Your fourth generated answer to the question\n"
              "Correct Answer: N \n" 
              "Explanation: Thorough explanation of why answer N is correct\n\n"
              "\n\Input_Paragraph: {}\\n").format(input_paragraph)

    # Generate the response
    response = model.generate_content(prompt)

    # Introduce a delay of 1 second to limit to 60 requests per minute
    time.sleep(1)

    # Return the text response
    #print(response.candidates)
    if len(response.candidates) > 0:
        response = response.candidates[0].content.parts[0].text.strip()
    else:
        response = response.text.strip()
    fields = response.split('\n')
    data = {}
    for field in fields:
        if 'Question' in field:
            value = field.split('Question:')[-1]
            key = 'Question'
        elif 'Answer 1' in field:
            value = field.split('Answer 1:')[-1]
            key = 'Answer 1'
        elif 'Answer 2' in field:
            value = field.split('Answer 2:')[-1]
            key = 'Answer 2'
        elif 'Answer 3' in field:
            value = field.split('Answer 3:')[-1]
            key = 'Answer 3'
        elif 'Answer 4' in field:
            value = field.split('Answer 4:')[-1]
            key = 'Answer 4'
        elif 'Correct Answer' in field:
            value = field.split('Correct Answer:')[-1]
            key = 'Correct Answer' 
        elif 'Explanation' in field:
            value = field.split('Explanation:')[-1]
            key = 'Explanation'
        #value = field.split(':')[0]
        data[key] = value.strip()
    return data 
#sample_p = "The parotid glands are situated anteriorly and inferiorly to the ear.They overlie the vertical mandibular rami and masseter muscles, behind which they extend into the retromandibular sulci.The glands extend superiorly from the zygomatic arches and inferiorly to below the angles of the mandible where they overlie the posterior bellies of the digastric and the sternocleidomastoid muscles.The parotid duct exits the gland anteriorly, crosses the masseter muscle, curves medially around its anterior margin, pierces the buccinator muscle, and enters the mouth opposite the 2 nd upper molar tooth."
#response = query_gemini(sample_p)
#print(response)
with open('results.jsonl', 'w') as f:
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        para = row['paragraph']
        response = query_gemini(para)
        #print(response)
        json.dump(response, f)
        f.write("\n")
    