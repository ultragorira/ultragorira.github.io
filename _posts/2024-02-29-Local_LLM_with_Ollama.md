# Create your own local and private LLM with Ollama 

The hype for LLM and Generative AI is not stopping at all and actually there is a continuous pushing of the boundaries of what's possible, see for example [Sora from OpenAI](https://openai.com/sora). That's just mind-blowing. 
In this post, though, I wanted to keep it short and simple. I started to experiment more LLMs and compare them, but one thing that I wanted was to have it running locally, privately, without paying any monthly fee. There are options available, like LLM Studio, but I wanted to build something simple myself. Here comes [Ollama](https://ollama.com/) to the rescue! 


## What is Ollama?

Ollama makes it very to run LLMs locally. Until January it was only available for MacOS and Linux but just recently, in February, they released the preview version for Windows as well. 

The installation is pretty straightforward. Just download it and install it on your machine, in my case, a Windows machine.
Once installed, you are free to download any of the models available. A list of all models can be found [here](https://ollama.com/library)

### Downloading a model

To download a model, use the terminal by typing the following command:

```
ollama pull [model]
```
For example, as my first model I download Mistral (7B). Depending on the hardware you have available, you can run smaller or bigger models. Based on the GitHub page, these are the RAM needs:

**You should have at least 8 GB of RAM available to run the 7B models, 16 GB to run the 13B models, and 32 GB to run the 33B models.**

Yes, you can run models also on CPU. However, having a GPU does make it a nicer experience, as Ollama does have a built-in GPU accelerator. 

### Inference to a model

Once you have downloaded the model, you can interact with it by running 

```
ollama run [model]
```

Note that Ollama supports also multimodal LLM, like Llava, e.g.

```
>>> What's in this image? /Users/jmorgan/Desktop/smile.png
The image features a yellow smiley face, which is likely the central focus of the picture.
```

Or even passing a file as argument

```
$ ollama run llama2 "Summarize this file: $(cat README.md)"
 Ollama is a lightweight, extensible framework for building and running language models on the local machine. It provides a simple API for creating, running, and managing models, as well as a library of pre-built models that can be easily used in a variety of applications.
```

That's just fantastic but maybe we can step it a notch and make it even nicer. What about having a GUI that mimics for example ChatGPT? This can be easily achieved with Gradio or Streamlit. Additionally Ollama has the [Python package](https://pypi.org/project/ollama/) which makes it very easy to create such app. 

So, let's implement this!


## Local LLM with Ollama and Streamlit

The code is really short and it's for this version only having text input but it could be extended to have file uploads or images. 

```

from typing import Generator
import ollama
import streamlit as st

st.set_page_config(
        page_title="Ollama Local",
        page_icon="🦙",
        layout="wide",
    )



def main() -> None:
    """
    Main function to run the Ollama chat application.
    """
    st.title("Local LLMs")
    st.image('ollama.png', width = 300)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "model" not in st.session_state:
        st.session_state["model"] = ""

    models = [model["name"] for model in ollama.list()["models"]
              if "name" in model]
    st.session_state["model"] = st.selectbox("Select model:", models)

    display_chat_messages()

    if prompt := st.chat_input("Type your prompt"):
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message = st.write_stream(infer_model())
            st.session_state["messages"].append({"role": "assistant", "content": message})

def display_chat_messages() -> None:
    """
    Display chat messages in the Streamlit app.
    """
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def infer_model() -> Generator[str, None, None]:
    """
    Generator function to retrieve model responses from Ollama chat.
    """
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

if __name__ == "__main__":
    main()

```

To run it just issue the cmd from within the folder where your app.py is 

```
streamlit run app.py
```

A localhost will spin up and the app will be available.
Note that in the dropdown you will see the models available within your system downloaded with Ollama. In my case I have Mistral (7b), Llava, Llama2 (7B) and Phi.

Let's see it in action!

<video src="https://github.com/ultragorira/ultragorira.github.io/assets/62200472/18004937-adfe-4710-a782-4b23bfe869d0" controls="controls" style="max-width: 730px;">
</video>




**Note that this machine has NVIDIA RTX 2070 Max-Q (about 4 years old).**

It's just crazy to think that just a year ago the only LLM available was ChatGPT and now you can do your own locally and privately. The future is bright!
