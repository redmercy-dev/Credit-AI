import json
import traceback
import nest_asyncio
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import streamlit as st
import asyncio
from typing import List, Tuple, Optional

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

openai_api_key = st.secrets["api_keys"]["openai_api_key"]
client = OpenAI(api_key=openai_api_key)

tools = [

    {"type": "code_interpreter"},
    {"type": "file_search"}
]

available_functions = {
}

instructions = """
You are an Assessment Document Generator whose primary function is to create detailed credit assessment reports for users based on the credit reports they upload. Begin by utilizing the file search functionality to locate and access the credit report provided by the user. Once accessed, extract all relevant data from the report, including personal identifiers such as the user's name and contact information, as well as credit information like the current credit score, risk category, credit inquiries, adverse listings, payment history, and credit utilization rates.

When generating the assessment report, adhere to the following structure to maintain consistency and clarity:

Assessment Report

Introduction: Start by greeting the recipient by their first name, which should be retrieved from the extracted data include the username do not use dear client . Introduce yourself as Mandy, Credit Analysis Assistant from Credit Fix, and acknowledge the user's membership status, highlighting any associated benefits. Clearly state the purpose of the report, which is to outline a personalized credit improvement plan.

Credit Report Summary: Present the user's current credit score and indicate the corresponding risk category, such as Excellent, Good, Fair, or Poor. Identify and detail the key issues affecting the user's credit, including the count and frequency of credit inquiries, details of any adverse listings like bankruptcies or charge-offs, any missed payments or accounts in arrears, and the current credit utilization percentage. Provide an explanation of how each of these key issues impacts the user's overall creditworthiness.

Personalized Action Plan: Offer tailored recommendations for improving the user's credit. Advise the user to follow the provided application strategies guide and recommend waiting six to twelve months before making new credit applications to enhance their credit standing. Suggest using a budget planning tool to manage finances effectively and setting up automated payments to ensure timely bill payments. Encourage the user to follow the optimization module to reduce their credit utilization rates.

Monitoring & Resources: Inform the user about the availability of ongoing monthly credit monitoring services to track their progress. Schedule a follow-up analysis after 90 days to assess improvements and make any necessary adjustments. Provide access to various tools, including credit report guides to help users understand their credit reports, dispute templates for addressing inaccuracies, debt settlement tactics, and financial planning tools for effective financial management.

Closing: Summarize the recommended action steps for the user to follow. If applicable, invite the user to join or renew their membership, emphasizing additional benefits. Provide contact information for further assistance, such as email addresses or phone numbers. Include relevant visual aids like charts and graphs to illustrate key points and ensure that all personal identifiers are appropriately used, as extracted from the user's uploaded credit report.

Additionally, leverage all available resources from Credit Fix to enhance the report's quality and usefulness. Include contact information for major credit bureaus to facilitate any necessary communications:

Experian: consumer@experian.co.za
TransUnion: legal@transunion.co.za
XDS: disputes@xds.co.za
Ensure the report is professionally formatted with clear headings and sections, using concise and user-friendly language. Incorporate visual elements where appropriate to enhance comprehension. By following these detailed instructions and maintaining the specified format, you will generate a thorough and personalized credit assessment report that effectively assists users in improving their credit profiles.




"""

def create_assistant(file_ids):
    vector_store = client.beta.vector_stores.create(
        name="Assistant Knowledge Base",
        file_ids=file_ids
    )

    assistant = client.beta.assistants.create(
        name="Credit Analysis Assistant",
        instructions=instructions,
        model="gpt-4o",
        tools=tools,
        tool_resources={
            'file_search': {
                'vector_store_ids': [vector_store.id]
            }
        }
    )
    return assistant.id

def handle_tool_outputs(run):
    tool_outputs = []
    try:
        for call in run.required_action.submit_tool_outputs.tool_calls:
            function_name = call.function.name
            function = available_functions.get(function_name)
            if not function:
                # Handle cases where the function is not available
                st.error(f"Function {function_name} is not available.")
                output = f"Function {function_name} is currently disabled."
            else:
                arguments = json.loads(call.function.arguments)
                with st.spinner(f"Executing {function_name}..."):
                    output = function(**arguments)
                    if output is None:
                        output = f"No content returned from {function_name}"
            tool_outputs.append({
                "tool_call_id": call.id,
                "output": json.dumps(output)
            })
        return client.beta.threads.runs.submit_tool_outputs(
            thread_id=st.session_state.user_thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
    except Exception as e:
        st.error(f"Error in handle_tool_outputs: {str(e)}")
        return None

def create_vector_store_for_file(file_id: str, name_prefix: str = "Message") -> Optional[str]:
    """
    Create a vector store for a single file with a 7-day expiration policy.
    Returns the vector store ID if successful, None otherwise.
    """
    try:
        vector_store = client.beta.vector_stores.create(
            name=f"{name_prefix}-{file_id}",
            expires_after={
                "anchor": "last_active_at",
                "days": 7
            }
        )
        
        # Create a batch with the single file and wait for processing
        batch = client.beta.vector_stores.file_batches.create_and_poll(
            vector_store_id=vector_store.id,
            file_ids=[file_id]
        )
        
        if batch.status == "completed":
            return vector_store.id
        else:
            st.error(f"Failed to process file in vector store: {batch.status}")
            return None
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

async def get_agent_response(assistant_id: str, user_message: str, file_id: Optional[str] = None) -> Tuple[str, List, List]:
    """
    Get response from the assistant, creating a new vector store for the uploaded file if present.
    """
    try:
        with st.spinner("Processing your request..."):
            # Create message attachments and vector store if file is present
            attachments = []
            if file_id:
                # Create a new vector store for this file
                vector_store_id = create_vector_store_for_file(file_id)
                if vector_store_id:
                    # Update thread with the new vector store
                    client.beta.threads.update(
                        thread_id=st.session_state.user_thread.id,
                        tool_resources={
                            "file_search": {
                                "vector_store_ids": [vector_store_id]
                            }
                        }
                    )
                    # Add file as attachment to the message
                    attachments = [{
                        "file_id": file_id,
                        "tools": [{"type": "file_search"}]
                    }]

            # Create the message with attachments
            client.beta.threads.messages.create(
                thread_id=st.session_state.user_thread.id,
                role="user",
                content=user_message,
                attachments=attachments
            )
            # Create run (without tool_resources)
            run = client.beta.threads.runs.create(
                thread_id=st.session_state.user_thread.id,
                assistant_id=assistant_id
            )

            while run.status in ["queued", "in_progress"]:
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.user_thread.id,
                    run_id=run.id
                )
                if run.status == "requires_action":
                    run = handle_tool_outputs(run)
                await asyncio.sleep(1)

            # Process response
            last_message = client.beta.threads.messages.list(
                thread_id=st.session_state.user_thread.id,
                limit=1
            ).data[0]

            formatted_response_text = ""
            download_links = []
            images = []

            if last_message.role == "assistant":
                for content in last_message.content:
                    if content.type == "text":
                        formatted_response_text += content.text.value
                        for annotation in content.text.annotations:
                            if annotation.type == "file_path":
                                file_id = annotation.file_path.file_id
                                file_name = annotation.text.split('/')[-1]
                                file_content = client.files.content(file_id).read()
                                download_links.append((file_name, file_content))
                    elif content.type == "image_file":
                        file_id = content.image_file.file_id
                        image_data = client.files.content(file_id).read()
                        images.append((f"{file_id}.png", image_data))
                        formatted_response_text += f"[Image generated: {file_id}.png]\n"

            return formatted_response_text, download_links, images

    except Exception as e:
        st.error(f"Error in get_agent_response: {str(e)}")
        return f"Error: {str(e)}", [], []
def main():
    st.title("Credit Analysis Assistant")
    st.sidebar.title("Assistant Configuration")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    assistant_choice = st.sidebar.radio("Choose an option:", ["Create New Assistant", "Use Existing Assistant"])

    if assistant_choice == "Create New Assistant":
        uploaded_files = st.sidebar.file_uploader(
            "Upload files for assistant (e.g., internal ebook)", 
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.sidebar.button("Create New Assistant"):
                file_ids = []
                for uploaded_file in uploaded_files:
                    file_info = client.files.create(file=uploaded_file, purpose='assistants')
                    file_ids.append(file_info.id)
                st.session_state.assistant_id = create_assistant(file_ids)
                st.sidebar.success(f"New assistant created with ID: {st.session_state.assistant_id}")
                st.session_state.user_thread = client.beta.threads.create()
                st.session_state.messages = []

    else:
        assistant_id = st.sidebar.text_input("Enter existing assistant ID:")
        if assistant_id:
            st.session_state.assistant_id = assistant_id
            if 'user_thread' not in st.session_state:
                st.session_state.user_thread = client.beta.threads.create()
            st.sidebar.success(f"Using assistant with ID: {assistant_id}")

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "downloads" in message:
                for file_idx, (file_name, file_content) in enumerate(message["downloads"]):
                    st.download_button(
                        label=f"Download {file_name}",
                        data=file_content,
                        file_name=file_name,
                        mime="application/octet-stream",
                        key=f"download_{idx}_{file_idx}"  # Added unique key
                    )
                    if file_name.endswith('.html'):
                        st.components.v1.html(file_content.decode(), height=300, scrolling=True)
            if "images" in message:
                for img_idx, (image_name, image_data) in enumerate(message["images"]):
                    st.image(image_data)
                    st.download_button(
                        label=f"Download {image_name}",
                        data=image_data,
                        file_name=image_name,
                        mime="image/png",
                        key=f"image_download_{idx}_{img_idx}"  # Added unique key
                    )

    # File upload handling
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf", "txt"])
    current_file_id = None
    
    if uploaded_file:
        try:
            file_info = client.files.create(file=uploaded_file, purpose='assistants')
            current_file_id = file_info.id
            st.success(f"File {uploaded_file.name} uploaded successfully!")
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

    # Chat input handling
    prompt = st.chat_input("You:")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if 'assistant_id' in st.session_state:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response, download_links, images = asyncio.run(
                    get_agent_response(st.session_state.assistant_id, prompt, current_file_id)
                )
                message_placeholder.markdown(response)
                
                # Add unique keys for current message downloads
                for idx, (file_name, file_content) in enumerate(download_links):
                    st.download_button(
                        label=f"Download {file_name}",
                        data=file_content,
                        file_name=file_name,
                        mime="application/octet-stream",
                        key=f"current_download_{len(st.session_state.messages)}_{idx}"
                    )
                    if file_name.endswith('.html'):
                        st.components.v1.html(file_content.decode(), height=300, scrolling=True)
                
                # Add unique keys for current message images
                for idx, (image_name, image_data) in enumerate(images):
                    st.image(image_data)
                    st.download_button(
                        label=f"Download {image_name}",
                        data=image_data,
                        file_name=image_name,
                        mime="image/png",
                        key=f"current_image_{len(st.session_state.messages)}_{idx}"
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "downloads": download_links,
                "images": images
            })
        else:
            st.warning("Please create a new assistant or enter an existing assistant ID before chatting.")


if __name__ == "__main__":
    main()
