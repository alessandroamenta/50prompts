import base64
import pandas as pd
import streamlit as st
import asyncio
import aiohttp
from aiohttp.client_exceptions import ContentTypeError


API_URL = "https://api.openai.com/v1/chat/completions"

# Sidebar
st.sidebar.title("🛠️ Settings")
API_KEY = st.sidebar.text_input("🔑 OpenAI API Key", value='', type='password')
model_choice = st.sidebar.selectbox("🤖 Choose model:", ["gpt-3.5-turbo-16k", "gpt-4"])

# Add a text area for common instructions in the sidebar
with st.sidebar.expander("📝 Custom Instructions"):
    common_instructions = st.text_area(
        "Enter instructions to apply to all prompts (e.g., 'You are an expert copywriter, respond in Dutch.')", 
        ''
    )

# Instructions Expander
with st.sidebar.expander("🔍 How to use"): 
    st.write("""
    1. 🔑 Input your OpenAI API key.
    2. 🤖 Pick the model.
    3. ✍️ Add custom instructions for all prompts (if needed).
    4. 📥 Choose the input method: Text Box or File Upload.
    5. 📝 If using Text Box, separate each prompt with a blank line.
    6. 📂 If using File Upload, upload a CSV or Excel file.
    7. 🚀 Click the "Generate Answers" button.
    8. 📤 Once answers are generated, download the CSV file with results.
    """)

st.title("🧠 GPT Answer Generator")
st.write("""
Generate answers for up to 50 prompts using OpenAI.
""")
st.warning("For best performance and to stay within OpenAI's rate limits, limit to 50 prompts.")


# Pricing details
PRICING = {
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    "gpt-4": {"input": 0.03, "output": 0.06}
}

async def get_answer(prompt, model_choice, common_instructions):
    # Prepend instructions to the actual prompt if provided
    full_prompt = f"{common_instructions}\n{prompt}" if common_instructions else prompt

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "OpenAI Python v0.27.3"
    }
    data = {
        "model": model_choice,
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 1250,
        "temperature": 0.3,
        "top_p": 1,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, json=data) as response:
                response_data = await response.json()
                return response_data['choices'][0]['message']['content']
    except ContentTypeError:
        st.error("There's an error on OpenAI's end. No worries, just try again chief, it should work on the next try! :)")
        return "Error: Couldn't fetch the answer for this prompt."

# Radio button to select input method
input_method = st.radio("📥 Choose input method:", ["Text Box", "File Upload"])

if input_method == "Text Box":
    st.write("Please separate each prompt with a blank line.")
    user_input = st.text_area("Enter up to 50 prompts:", height=300)
    prompts = user_input.split('\n\n')  # Split by two newlines

elif input_method == "File Upload":
    uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        prompts = df.iloc[0:, 0].tolist()  # Read prompts from the first column

else:
    prompts = []

# Modify the button click event to pass the common instructions to the get_answer function
if st.button("🚀 Generate Answers"):
    with st.spinner('👩‍🍳 GPT is whipping up your answers! Hang tight, this will just take a moment... 🍳'):
        answers = []

        # Use asyncio to process the prompts concurrently
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        answers = loop.run_until_complete(asyncio.gather(*(get_answer(prompt, model_choice, common_instructions) for prompt in prompts)))

        # Calculate tokens (this method is a rough estimate; adjust as needed)
        total_tokens = sum(len(prompt.split()) + len(answer.split()) for prompt, answer in zip(prompts, answers))

        # Calculate the cost based on model choice
        input_cost = (total_tokens / 1000) * PRICING[model_choice]["input"]
        output_cost = (total_tokens / 1000) * PRICING[model_choice]["output"]
        total_cost = input_cost + output_cost

        # Display the total tokens used and the cost
        st.write(f"Total tokens used: {total_tokens}")
        st.write(f"Total cost: ${total_cost:.2f}")

        # Create a DataFrame
        df = pd.DataFrame({
            'Prompts': prompts,
            'Answers': answers
        })

        # Convert DataFrame to CSV and let the user download it
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        
        st.success("🎉 Answers generated successfully!")
        
        # Display the styled download link directly
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="answers.csv" style="display: inline-block; padding: 0.25em 0.5em; text-decoration: none; background-color: #4CAF50; color: white; border-radius: 3px; cursor: pointer;">📤 Download CSV File</a>', unsafe_allow_html=True)
