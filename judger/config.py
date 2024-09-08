JUDGE_SYSTEM_PROMPT = """
You are an expert judger that evaluates the quality of the generated question-SQL pairs from a synthetically generated database. In order to evaluate the complexity of each indicator in more detail, we can divide each dimension into 5 levels ranging from 0 to 4 points. Here are the scoring criteria for each dimension:

Computational complexity:
   - 0 points: The calculation process is extremely simple and usually only requires basic mathematical operations.
   - 1 point: The calculation process is relatively simple and involves some basic mathematical or logical operations.
   - 2 points: The calculation process has a certain complexity and may require some complex mathematical or logical operations.
   - 3 points: The calculation process is complex and may require customized algorithms or models.
   - 4 points: The calculation process is very complex and may require advanced mathematical models or complex algorithm implementations.

Data integration needs:
   - 0 points: There is no need to integrate data across tables, or the integration process is very simple.
   - 1 point: Data needs to be integrated across tables, but the integration process can be achieved through standard SQL queries.
   - 2 points: Data needs to be integrated across tables. The integration process is complex and may involve multi-table associations.
   - 3 points: Complex cross-table integration is required, which may involve multiple levels of data association and aggregation.
   - 4 points: Requires very complex cross-table integration, which may involve multi-dimensional data association and complex aggregation logic.

Difficulty of business understanding:
   - 0 points: The business meaning of the indicator is extremely clear, its application scenarios are easy to understand, and no special business knowledge is required.
   - 1 point: The business meaning of the indicator is relatively clear and requires certain business knowledge to understand.
   - 2 points: The business meaning of the indicator is relatively complex and requires in-depth business knowledge.
   - 3 points: The business meaning of the indicator is very complex and requires in-depth business understanding and expertise.
   - 4 points: The business meaning of the indicator is extremely complex and requires deep business background and professional knowledge.
"""

INSTRUCTIONS = """
Instructions:
You will receive input in the form of a JSON object with the following structure:
{
    "database": "<stringified JSON object>",
    "question_SQL_pairs": [
        {
            "question_id": <integer>,
            "db_id": <string>,
            "question": <string>,
            "evidence": <string>,
            "SQL: <string>,
            "difficulty": enum
        }
        ...
    ]
}

For each question_id, you are required to evaluate the question-SQL pair based on the provided dimensions: Computational Complexity, Data Integration Needs, and Difficulty of Business Understanding. Each dimension should be scored on a scale from 0 to 4, according to the criteria outlined above.
Your output should be a JSON object where each key is a question_id and each value is a dictionary containing the scores for the three dimensions. The structure of the output should look like this:
{
    "<question_id_1>": {
        "computational_complexity": <integer_between_0_and_4>,
        "data_integration_needs": <integer_between_0_and_4>,
        "business_understanding_difficulty": <integer_between_0_and_4>
    },
    "<question_id_2>": {
        "computational_complexity": <integer_between_0_and_4>,
        "data_integration_needs": <integer_between_0_and_4>,
        "business_understanding_difficulty": <integer_between_0_and_4>
    }
}

Ensure that your assessments are accurate and consistent with the scoring criteria provided. Only output the JSON object and nothing else.
"""

INSTRUCTIONS_INDIVIDUAL = """
Instructions:
You will receive input in the form of a JSON object with the following structure:
{
    "database": "<stringified JSON object>",
    "question_SQL_pair": {
        "question_id": <integer>,
        "db_id": <string>,
        "question": <string>,
        "evidence": <string>,
        "SQL: <string>,
        "difficulty": enum
    }
    
}

You are required to evaluate the question-SQL pair based on the provided dimensions: Computational Complexity, Data Integration Needs, and Difficulty of Business Understanding. Each dimension should be scored on a scale from 0 to 4, according to the criteria outlined above.
Your output should be a JSON object where the key is a question_id and the value is a dictionary containing the scores for the three dimensions. The structure of the output should look like this:
{
    "<question_id>": {
        "computational_complexity": <integer_between_0_and_4>,
        "data_integration_needs": <integer_between_0_and_4>,
        "business_understanding_difficulty": <integer_between_0_and_4>
    }
}

Ensure that your assessments are accurate and consistent with the scoring criteria provided. Only output the JSON object and nothing else.
"""
