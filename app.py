from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.vectorstores import Chroma

from transformers import (AutoTokenizer,
                           pipeline)
from auto_gptq import AutoGPTQForCausalLM

from constant import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
)
from prompts import prompt_template
import logging

import torch

logging.info(f"Cuda:: {torch.cuda.is_available()}")

#-----------------------------------------------------------------------------------------------------------------
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 4}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
      model_id="TheBloke/Llama-2-7b-Chat-GPTQ"
      model_basename="model.safetensors"
      # Remove the ".safetensors" ending if present
      model_basename = model_basename.replace(".safetensors", "")

      tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
      logging.info("Tokenizer loaded")

      model = AutoGPTQForCausalLM.from_quantized(
          model_id,
          model_basename=model_basename,
          use_safetensors=True,
          trust_remote_code=True,
          device="cuda:0",
          use_triton=False,
          quantize_config=None,
      )
      print("*** Pipeline:")
      pipe = pipeline(
          "text-generation",
          model=model,
          tokenizer=tokenizer,
          max_new_tokens=512,
          temperature=0.7,
          top_p=0.95,
          repetition_penalty=1.15
      )

      hug_model=HuggingFacePipeline(pipeline=pipe)
      return hug_model

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                       model_kwargs={'device': 'cuda'})
    # embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME,
    #                                    model_kwargs={'device': 'cpu'})
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Affine Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    # print('*'*70)
    # print("ANSWER::",answer)
    # print('*'*70)
    # sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    await cl.Message(content=answer).send()

