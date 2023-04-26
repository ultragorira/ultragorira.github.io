# Langchain and Streamlit 

[Langchain](https://python.langchain.com/en/latest/index.html) is probably the hottest framework right now if you are interested in building LLMs powered apps. So I thought, why not trying it out and combine it with some UI like Streamlit. 
There is quite a lot you can accomplish with Langchain. 

# Use Cases

For this blog post I wanted to focus on two use cases. 
ChatGPT and all the other available LLMs that are mushrooming almost on a daily basis do not really have access to the web, so as in the case of ChatGPT, there is a cut off date of September 2021. This means that anything that happened after that, ChatGPT does not really have knowledge of. There will be plug-ins available for ChatGPT that will allow it to access data from the internet but for now it is not really possible. Moreover, say you are a company that has some internal documentation, lots of it, and you want to be able to interact with the information in the documentation, without scanning through it, wouldn't it be cool to be able to have a ChatGPT-look-alike UI where you can ask about the documentation? Well, with Langchain you can create such thing, a customized GPT.

But that's not all. Langchain has so many features that it would be difficult to cover them all in a single post. However, one thing that I wanted to try out besides a customized GPT, was to be able to interact with data, with Excel data. 
If you like me work with data, manipulating and extracting insights with either pandas or polars, interacting with tabular data should not be a problem. However, in Langchain there are Agents, a series of them, that have specific skills. One of the available agents is called "create_csv_agent". This agent, based on the [documentation](https://python.langchain.com/en/latest/modules/agents/toolkits/examples/csv.html) lets you interact with a CSV. Under the hood this agent calls the Pandas DataFrame agent to do the data manipulation needed to reply to your query. It is actually pretty neat and lets people without deep knowledge of Excel or data manipulation to extract valuable insights with just natural language. So let's dive in. 

# CSV AGENT

The Streamlit app is very simple and has on the left side panel the options to interact either with the CSV Agent or the custom GPT. 

To run a query on a CSV is actually very simple and does not require that many lines of code. One thing to note is that you will need to have the OpenAI API Key in your environment variables, else this will not work. 

All you have to do, once you load the Excel in memory, is to pass the file to the agent, set the temperature (preferably to 0 so it does not make up information, although I will show you that in one case it gave an absurd answer despite the temperature set to 0) and the verbose to either True, if you want to see the whole chain on thoughts of the agent in the terminal or False if you do not. 

Here is a snapshot of the code to do that. 


```

    uploaded_file = st.file_uploader("Upload an Excel file")

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        st.write(df)
       
        agent = create_csv_agent(OpenAI(temperature=0),
                            uploaded_file.name,
                            verbose = True)
        
        question = st.text_area("Type your question here", height = 100)

        # Display the user's question
        if question:
            bot_out = agent.run(question)
            st.write(bot_out)

```

For this demo, I took a taxi dataset from Kaggle from [here](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

## See the CSV Agent in action




