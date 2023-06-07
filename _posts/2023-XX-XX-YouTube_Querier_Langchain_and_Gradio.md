# YouTube Querier with Langchain and Gradio

Following up on my latest blog post on [Langchain](https://ultragorira.github.io/2023/04/27/Langchain.html), in this short blog post I will show one of the possible ways to implement a YouTube Querier so that we can ask questions about a video. The idea is pretty basic but effective and does not require a lot of code, thanks to the Langchain framework and the various models and packages I am going to use for this scenario. 

## What we need

***Langchain*** to create embeddings (again OpenAI), create a vector store (Chroma) and interaction with it via LLM.

***PyTube*** to download the audio of a video from YouTube. Theoretically you could have any audio and create embeddings on it. 

***Whisper*** from OpenAI to create transcriptions on the downloaded audio.

***Gradio*** to easily create a chat UI. 

## Workflow

- Download audio from YouTube via PyTube.
- Pass the downloaded audio to Whisper
- Create from the segments returned from Whisper a corpus
- Split text with Langchain and create embeddings and save to vector store as a persist store so that we can read from the db any time and no need to recompute every time.
- Create UI in Gradio where we will ask questions to the LLM about the video