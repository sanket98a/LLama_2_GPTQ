

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, new_system_prompt ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template.strip()

# PROMPT-1 -------------------------------------------------------------------------------------------------------
# system_prompt = "You are an advanced assistant to help user to find the answer from the provided context.Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\ncontext:{context}"
# instruction = """user_question: {question}"""
# prompt_template=get_prompt(instruction,system_prompt)
# print(prompt_template)

# PROMPT-2 ----------------------------------------------------------------------------------------------------
system_prompt = "Use the following pieces of context to answer the question at the end. If the answer cannot be found, respond with 'The answer is not available in the given data'.\n"
instruction = """context:{context}, \n user_question: {question}"""
prompt_template=get_prompt(instruction,system_prompt)
print(prompt_template)

# PROMPT-3 ------------------------------------------------------------------------------------------------------
# prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Context: {context}
# Question: {question}
# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# PROMPT-4 -------------------------------------------------------------------------------------------------------
# prompt_template="""[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Use the following pieces of information to answer the user's question.If you don't know the answer, just say that you don't know, don't try to make up an answer.please return answer only from below context.
#     context:{context} \n<</SYS>>\n\n
#     question:{question}[/INST]
#     """.strip()
