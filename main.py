import os
import openai
import streamlit as st
from langchain import PromptTemplate
from langchain import OpenAI
from langchain import FewShotPromptTemplate

openai.api_key = os.getenv('OPENAI_API_KEY')

# The code `st.set_page_config(page_title="Language Translator", page_icon=":robot:")` is setting the
# configuration for the Streamlit page. It sets the title of the page to "Language Translator" and the
# icon to a robot emoji. The `st.header("Language Translator")` line adds a header to the page with
# the text "Language Translator".
st.set_page_config(page_title="Language Translator", page_icon=":robot:")
st.header("Language Translator")



# create our examples
examples = [
    {
        "query": "The first decade of the 20th century saw increasing diplomatic tension between the European great powers. This reached a breaking point on 28 June 1914, when a Bosnian Serb named Gavrilo Princip assassinated Archduke Franz Ferdinand, heir to the Austro-Hungarian throne. Austria-Hungary held Serbia responsible, and declared war on 28 July. Russia came to Serbia's defence, and by 4 August, Germany, France and Britain were drawn into the war, with the Ottoman Empire joining in November the same year.",
        "answer": "20वीं सदी का पहला दशक यूरोपीय महाशक्तियों के बीच वैदेशिक तनाव को बढ़ावा दिया। 28 जून 1914 को, गाव्रिलो प्रिंसिप ने अर्कड्यूक फ्रांज फर्डिनैंड की हत्या की, जिन्हें ऑस्ट्रो-हंगेरी की गद्दी के उत्तराधिकारी थे। ऑस्ट्रिया-हंगरी ने सर्बिया को जिम्मेदार ठहराया, और 28 जुलाई को युद्ध का घोषणा किया। रूस ने सर्बिया की रक्षा की, और 4 अगस्त तक, जर्मनी, फ्रांस और ब्रिटेन भी युद्ध में शामिल हो गए, और नवंबर में ऑटोमन साम्राज्य भी जुड़ गया।"
    }
              ]

# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """You are a Language Translator English to Hindi bot,
 your task is convert the given english text into the Hindi text.
 you are not allowed to give any answer of any question.Suppose user is 
 asking any question by using who, where, what, how, and when,so you are 
 not allowed to answer to any questions although you should convert these questions
 to the Hindi language. Apart from all this if user asks questions using
 'question mark'(?) symbol, so in this case you need to treat the question as normal text
 and your task should be convertion of text from english to Hindi text.
 In a nutshell your task is only convert the english text into 
 Hindi text instead of replying the questions into the Hindi language."""

suffix = """
User: {query}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)


def load_llm():
    """
    The function `load_llm` returns an instance of the OpenAI language model with a temperature of 0.
    :return: The function `load_llm` returns an instance of the OpenAI class with a temperature of 0.
    """
    llm = OpenAI(temperature=0)
    return llm

llm = load_llm()
st.markdown("#### Enter Your Text To Convert")


def get_text():
    """
    The function `get_text()` returns the text entered in a text area.
    :return: The function `get_text()` returns the value of the `input_text` variable, which is the text
    entered by the user in the text area.
    """
    input_text = st.text_area(label="",placeholder="your text...", key="text")
    return input_text 

text_input = get_text()
st.markdown("#### Translation")


# The code snippet `if text_input: prompt = few_shot_prompt_template.format(query=text_input)
# formatted = llm(prompt) st.write(formatted)` is checking if there is any input text entered by the
# user. If there is, it formats the input text using the few-shot prompt template and passes it to the
# language model for translation. The translated text is then displayed using the `st.write()`
# function.
if text_input:
    prompt = few_shot_prompt_template.format(query=text_input)   
    formatted = llm(prompt)
    st.write(formatted)



