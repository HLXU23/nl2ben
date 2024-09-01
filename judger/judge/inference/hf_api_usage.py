import math
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

load_dotenv()

MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'

client = InferenceClient(
    MODEL_ID,
    token=os.environ.get('HF_API_KEY'),
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

prompt = """Filename: POSB_Community_Outreach_&_Financial_Literacy_Initiatives.pdf
Information:
, we have raised closed to S$13 million  to date, which has
supported some 869,000 children , including beneﬁciaries from low-income families,
children with disabilities, and youths at risk, through more than 229 programmes  that
contribute towards social capital and community development.
“Money plays an important role in our lives as we need money in our daily
activities. Hence, saving is an important habit to cultivate, helping us to grow
money and increasing our buying power to get things that we dream for. The
vouchers given were rewards towards my savings. I used it to buy groceries
with my family and to get things that I want such as new stationeries and IT
gadget. In conclusion, I am eager to save money to reach my goals now!”
Jervez Neo, 16 years oldLogin
May I help you?
5/24/24, 3:47 PM We are all neighbours | POSB Singapore
https://www.posb.com.sg/personal/community 4/7        
POSB P Assion Run F or Kids 2023 POSB P Assion Run F or Kids 2023
Hear from our beneﬁciaries
Supporting Student Development
Carepack Beneﬁciary
“The carepack came in timely and I’m grateful for items such as the hand
sanitizer and masks as most were sold out outside. Many of the items were
also useful in my day-today life during this period where I am doing a lot of
home-based learning.”
Teo Wei Shan, 15 years old
Login
May I help you?
5/24/24, 3:47 PM We are all neighbours | POSB Singapore
https://www.posb.com.sg/personal/community 5/7        
POSB Heritage
 Check out the POSB’s Heritage here  (/iwov-resources/media/pdf/others/rfk-heritage-
2021.pdf?pid=sg-posb-pweb-others-rfk-textlink-posb-heritage-pdf)
Past Archives
2019
Was this information useful? Yes No
Temasek Polytechnic Graduand:
“I want to express my gr

Filename: OCBC_Youth_Financial_Empowerment_Program.pdfInformation:
At OCBC, we have raised close to S$15 million to date, which has supported over 900,000 young individuals, including those from low-income families and underprivileged backgrounds, through more than 250 programs aimed at promoting financial literacy and independence. Here's what one of our beneficiaries had to say:        

"I used to think that managing money was difficult and overwhelming, but the OCBC Youth Financial Empowerment Program taught me practical skills to help me make informed decisions about my finances. I learned how to 
create a budget, save for the future, and even invest 
in stocks. The program has given me the confidence to 
take control of my financial situation and plan for my future." - Emily Chan, 17 years old

At OCBC, we believe that financial literacy is a crucial life skill that can empower young people to make informed decisions about their futures. That's why we're committed to providing the necessary resources and support to help them achieve their financial goals.     

Login
May I help you?
5/24/24, 3:47 PM Empowering Youth | OCBC Bank
<https://www.ocbc.com.sg/personal-banking/youth/financial-empowerment-program.html>
Was this information useful? Yes No

Question: "1. How has POSB contributed to the community through their outreach and financial literacy initiatives?"
"""

completion = client.chat_completion(
	messages=[
        {'role': 'system', 'content': "Some information is retrieved from the database as provided based on the user's question. The assistant is to answer the question to the best of his/her ability, using only the information provided. The assistant must not add his/her own knowledge."},
        {'role': 'user', 'content': prompt}
    ],
	max_tokens=4096,
	stream=False,
)
answer = completion.choices[0].message.content
print(answer)
tokens = tokenizer(answer, return_tensors='pt', add_special_tokens=True)['input_ids']
shape = tokens.shape
print(shape)
print(math.prod(shape))
