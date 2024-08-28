# LLM-PTSD
<p>This repository contains the code and resources used to extract, preprocess, and analyze data from PTSD-related subreddits. The project is focused on creating a Large Language Model (LLM) designed to converse with PTSD patients, to provide a supportive and understanding environment for individuals dealing with PTSD</p>
<h2>Data Extraction</h2>
<p>The data used in this project was extracted from various subreddits where individuals share their experiences and challenges related to PTSD. These subreddits serve as a community space for people to discuss their symptoms, coping strategies, therapy experiences, and personal stories.</p>
<h2>Data Preprocessing</h2>
<p>The raw data underwent extensive preprocessing to prepare it for analysis and model training. This included:</p>
<ul><b>Text Cleaning</b><p>Remove special characters, links, and contractions to standardize the text.</p></ul>
<ul><b>Lemmatization</b><p>Conversion of words to their base forms reduces dimensionality and improves the model's generalization ability.</p></ul>
<ul><b>Spelling Correction</b><p>Automated correction of common spelling errors to ensure consistency across the dataset.</p></ul>
<h2>Data Anonymization</h2>
<p>To protect the privacy of individuals who shared their experiences, all personal identifiers were removed from the data. This was achieved using the 'scrubadub' module to anonymize the text, ensuring that the dataset is safe to use for research purposes</p>
<h2>Fine Tuning using Low-Rank Adaption and Differential Privacy</h2>
<p>The fine-tuning process employed LoRA to adapt the Llama-2-7b-chat model. LoRA allows the model to learn additional tasks while keeping the base model weights frozen, making the training process more efficient and less resource-intensive. This technique was beneficial given the large size of the Llama-2-7b model.To further ensure the privacy and confidentiality of the data, differential privacy techniques  were incorporated during the fine-tuning process. The private details were hidden by adding fake data in their place. This helped prevent the model from memorizing and reproducing specific details from the training data, thus protecting user anonymity.</p>
<h2>Integration of EHR Data</h2>
<p>The fine-tuned model was also integrated with Electronic Health Records (EHR) data to enhance its understanding of medical histories, treatment regimens, and patient demographics. This integration allowed the model to provide more personalized and accurate responses, particularly in conversations that involve discussing treatment options and progress.</p>
