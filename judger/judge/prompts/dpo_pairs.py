from typing import TypedDict

from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class OpenAIDPOResponse(TypedDict):
    # question: str
    answer: str


DEFAULT_SYSTEM_MESSAGE = (
    "You are an AI Data Scientist who creates high quality datasets that can be used for fine-tuning of Large Language Models."
    " Follow the user's instruction closely to create a dataset based on the given context."
)

teacher_system_prompt_v1: ChatCompletionSystemMessageParam = {
    'role': 'system',
    'content': DEFAULT_SYSTEM_MESSAGE,
}

student_system_prompt_v1 = teacher_system_prompt_v1


def teacher_user_prompt_v1(
    question: str,
    filename: str,
    content: str
) -> ChatCompletionUserMessageParam:
    prompt = f"""You are to read the following information and answer the user's question.
Filename: {filename}
Information: {content}

Now, answer the question, and you may elaborate as necessary. Do not create information that is not found in the information provided.
Question: "{question}"

You will reply in the following JSON format:
{{
    'answer': "<Answer>"
}}"""

    return {
        'role': 'user',
        'content': prompt,
    }


def student_user_prompt_v1(
    question: str,
    filename1: str,
    content1: str,
    filename2: str,
    content2: str
) -> ChatCompletionUserMessageParam:
    prompt = f"""You are to read the following information and answer the user's question.
Filename: {filename1}
Information: {content1}

Filename: {filename2}
Information: {content2}

Now, answer the question, and you may elaborate as necessary. Do not create information that is not found in the information provided.
Question: {question}"""

    return {
        'role': 'user',
        'content': prompt,
    }
