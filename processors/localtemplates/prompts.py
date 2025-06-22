MAIN_PROMPT = """
You are a copywriter and you have been provided with the following paragraph:
{context}
Using the paragraph above, 
generate 3 questions about the topics discussed in the paragraph, 
and provide an accurate answer for each question. Do not stray from the information provided in the paragraph.  
Try not to exceed 2 sentences for each answer.
Write the output as a JSON list.
"""