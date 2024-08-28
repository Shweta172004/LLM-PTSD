import scrubadub
import pandas as pd
# Read the text file
input_file_path =  'cleaned_PTSD_data.txt'
output_file_path = 'anonymized_text_file.txt'

with open(input_file_path, 'r') as file:
    text = file.read()

# Anonymize the text
anonymized_text = scrubadub.clean(text)

# Save the anonymized text to a new file
with open(output_file_path, 'w') as file:
    file.write(anonymized_text)

print(f"Anonymized text saved to {output_file_path}")
