#!/bin/bash

mkdir -p ../results

# List of database and corresponding qsql pairs files
declare -a database_files=("california_schools/california_schools.sqlite" "card_games/card_games.sqlite" "codebase_community/codebase_community.sqlite" "debit_card_specializing/debit_card_specializing.sqlite" "european_football_2/european_football_2.sqlite" "financial/financial.sqlite" "formula_1/formula_1.sqlite" "student_club/student_club.sqlite" "superhero/superhero.sqlite" "thrombosis_prediction/thrombosis_prediction.sqlite" "toxicology/toxicology.sqlite")
declare -a qsql_pairs_files=("california_schools/questions.json" "card_games/questions.json" "codebase_community/questions.json" "debit_card_specializing/questions.json" "european_football_2/questions.json" "financial/questions.json" "formula_1/questions.json" "student_club/questions.json" "superhero/questions.json" "thrombosis_prediction/questions.json" "toxicology/questions.json")

# Ensure both arrays have the same length
if [ ${#database_files[@]} -ne ${#qsql_pairs_files[@]} ]; then
    echo "Error: The number of database files and qsql_pairs files do not match."
    exit 1
fi

# Iterate over the files and run the commands
for i in "${!database_files[@]}"; do
    database_file="${database_files[$i]}"
    qsql_pairs_file="${qsql_pairs_files[$i]}"
    
    # Get the base names without extensions
    db_name=$(basename "$database_file" .sqlite) # Extracts db1 from folder/db1.sqlite
    qsql_name=$(basename "$qsql_pairs_file" .json) # Extracts qsql1 from qsql1.json
    
    # Define the output path as <database_name>_<qsql_name>.json
    output_file="../results/${db_name}_${qsql_name}.json"
    
    # Run the judge.py script
    python ../judge.py --database_file "$database_file" --qsql_pairs_file "$qsql_pairs_file" --evaluator deepseek-chat --output_path "$output_file" --distinct True
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Command failed for $database_file and $qsql_pairs_file"
    else
        echo "Output written to $output_file"
    fi
done

echo "All commands executed."
