import simple_question_left_questions_generator as simple_question_left
import questions_generator

import json
import random
import copy
import pandas as pd
import sqlite3
import time
from itertools import takewhile, repeat
from typing import List
from pathlib import Path

# Get the number of lines from the text file given in input
def rawincount(filename):
    with open(filename, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
        return sum( buf.count(b'\n') for buf in bufgen )

# Extract only base Wikipedia entities embeddings, excluding those of other languages. Since I had problems on file appending (an [Errno 5] input/output error),
# it's possible to build separate files adding the name of existing files into "existing_files" and the name of the output file into
# "output_file". In addition if that error happens this operation is done automatically. Once all files are created, they have to be merged into one file
def embeddings_filter(existing_files: List[str] = None, output_file: str = "wikidata_translation_v1_filtered.tsv"):
    if existing_files is None:
        existing_files = []
    original_output_file = output_file
    with open("models/wikidata_translation_v1.tsv", "r") as embeddings_file:
        lines_skipped = 0
        lines_number = 0
        for existing_file in existing_files:
            lines_number += rawincount("models/" + existing_file)
            print("Lines number: " + str(lines_number))
        for line in embeddings_file:
            if lines_skipped < lines_number and line[0:6] == "<http:":
                if lines_skipped % 100000 == 0:
                    print(lines_skipped)
                lines_skipped += 1
            elif line[0:6] == "<http:":
                try:
                    with open("models/" + output_file, "a") as filtered_embeddings_file:
                        filtered_embeddings_file.write(line)
                except IOError:
                    # Create a new file because it's not possible to use the old any more
                    output_file_dot_index = output_file.rindex(".")
                    output_file_underscore_index = output_file.rindex("_", 0, output_file_dot_index)
                    output_file_version_substring = output_file[output_file_underscore_index + 1 : output_file_dot_index]
                    if output_file_version_substring.isnumeric():
                        output_file_version_number = int(output_file_version_substring) + 1
                        output_file = output_file[:output_file_underscore_index + 1] + str(output_file_version_number) + output_file[output_file_dot_index:]
                    else:
                        output_file = output_file[:output_file_dot_index] + "_2" + output_file[output_file_dot_index:]
                    with open("models/" + output_file, "a") as filtered_embeddings_file:
                        filtered_embeddings_file.write(line)
                    continue

def tsv_to_sql(files_list: List[str], chunksize: int):
    # Create DB if not exist
    db_already_created = True
    if not Path('models/dataset_embeddings.db').is_file():
        Path('models/dataset_embeddings.db').touch()
        db_already_created = False
    # Connect to DB
    conn = sqlite3.connect('models/dataset_embeddings.db')
    c = conn.cursor()
    if not db_already_created:
        query = "CREATE TABLE TransE (qid text PRIMARY KEY ON CONFLICT IGNORE"
        for i in range(200):
            query += ", embedding" + str(i) + " int"
        query += ")"
        c.execute(query)
    columns = ["qid"]
    for i in range(200):
        columns.append("embedding" + str(i))
    for file_element in files_list:
        print("Loading file " + file_element + "...")
        chunk_number = 0
        # Load the data into a Pandas DataFrame, using chunks to handle smaller quantities of values
        for chunk in pd.read_csv("models/" + file_element, sep='\t', names = columns, header=None, chunksize=chunksize):
            print("Reading " + str(chunksize) + " elements...")
            for i in range(chunksize):
                # The last chunk won't be complete
                if i + chunk_number*chunksize >= chunk.size:
                    break
                chunk.at[i + chunk_number*chunksize, 'qid'] = chunk.at[i + chunk_number*chunksize, 'qid'].split("/")[-1].split(">")[0]
            # Write the data to a sqlite table
            chunk.to_sql('TransE', conn, if_exists='append', index = False)
            chunk_number += 1
    c.close()

def test_sql():
    conn = sqlite3.connect('models/dataset_embeddings.db')
    c = conn.cursor()
    results = c.execute("SELECT * FROM TransE WHERE qid='Q14859382'").fetchone()
    print(results)
    c.close()

start_time = time.time()

embeddings_filter()
tsv_to_sql(["wikidata_translation_v1_filtered.tsv", "wikidata_translation_v1_filtered_2.tsv", "wikidata_translation_v1_filtered_3.tsv"], 10**5)

test_sql()

print("--- %s seconds ---" % (time.time() - start_time))