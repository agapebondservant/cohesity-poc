MAIN_PROMPT = """
You are a copywriter and you have been provided with the following paragraph:
{context}
Using the paragraph above, 
generate 3 questions about the topics discussed in the paragraph, 
and provide an accurate answer for each question. Focus only on the content in the paragraph.  
Write the output as a JSON list.
"""