from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import fitz  # PyMuPDF
import google.generativeai as genai
import pandas as pd
import io
import re
from datetime import datetime

load_dotenv()  # take environment variables from .env.

# Configure the Gemini model with the API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Function to load Gemini model and get responses
def get_gemini_response(input_text, image_parts, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
    response = model.generate_content([input_text, image_parts[0], prompt])
    if response.candidates:
        return response.candidates[0].content
    else:
        return ""

# Function to prepare image for Gemini model
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to convert PDF to image
def pdf_to_image(uploaded_file):
    # Save the uploaded PDF file to a temporary location
    temp_pdf_path = "temp_uploaded_file.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    doc = fitz.open(temp_pdf_path)
    page = doc.load_page(0)  # Always use the first page
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Close the document before deleting the temporary file
    doc.close()
    os.remove(temp_pdf_path)

    return img   
 
def extract_substrings(text):
    # Define the pattern to match substrings starting from '{' and ending at '}'
    pattern = re.compile(r'\{(.*?)\}', re.DOTALL)

    # Find all substrings matching the pattern
    matches = pattern.findall(text)

    # Strip any leading/trailing whitespace from the substrings
    stripped_matches = [match.strip() for match in matches]

    return stripped_matches

def text_to_dataframe(text):
    # Remove the 'text: "' and trailing '"\n' from the input
    text = re.split(r'text: "', text)[1]
    text = re.split(r'"\|\s*\\n', text)[0]

    # Split the input text into lines using regex to match either '| \\n' or '|\\n'
    lines = re.split(r'\|\s*\\n', text.strip())

    # Extract the header and the data lines
    header = [col.strip() for col in lines[0].split('| ')]
    data = []

    for line in lines[1:]:
        if line.strip():  # Ignore empty lines
            data.append([item.strip() for item in line.split('| ')])

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data, columns=header)

    # Drop the last row from the DataFrame
    if not df.empty:
        df = df[:-1]

    return df


# Initialize Streamlit app
st.set_page_config(page_title="Invoice PDF to excel")

# Custom CSS for grey background
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("Invoice PDF to excel")
uploaded_files = st.file_uploader("Choose images or PDF files: ", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

st.write(f"Number of uploaded files: {len(uploaded_files)}")

extraction_prompt = """You are an expert in understanding invoices.
You will receive input images as invoices and answer questions based on them.
Ensure you include every relevant name regardless of how they are described (e.g., items, instruments, products).
If you did not find any answer for the product name, use Description of goods to find the product name whatever name is there.
Ensure that for finding each product or Description of goods, either a Product ID or Serial Number or product code exists that relates to the product or goods mentioned in the invoice.
Avoid unnecessary items being taken as product names and avoid repeating product names. Only consider the product name if an Item ID or Serial Number or product id is given of that product given.
Do not provide either extra product names or Description of goods beyond what is mentioned in the invoice.
please make sure that Ensure that for finding each product or Description of goods, either HSN number or number of quantity  mentioned in the invoice.
To ensure that the "Per Unit Price" is returned as just the rate without the unit of quantity, and that the "Product Name" and "Quantity with unit" are not combined,
Ensure that quantity, per unit price and total amount is numeric not text.
please do not repeat the same product name twice and avoid unnecessary names being taken as product names.
please make sure that Ensure that for finding each product or Description of goods, either HSN number  mentioned in the invoice.
please exclude those items whose HSN number is not given.
Please follow this rule correctly for all invoices, and ensure that each field is placed in its designated column without merging with other fields.
Please fill up the invoice number, supplier name, and bill-to details for different items, not leaving them blank after the first item.
please make sure that Always return all responses in this structural format: Invoice No| Supplier Name| Bill To Details| Product Name| Quantity| Unit of measurement| Per Unit Price| Amount| . and ensure that each field is placed in its designated column without merging with other fields and maintain this structure properly.
Please make sure there are seven fields.
please make sure all response text first line contain all fields.
"""

# Set a fixed input prompt
input_text = "give me the Invoice No, Supplier Name,  Bill To Details, Product Name, Quantity, Unit of measurement, Per Unit Price, Amount"
submit = st.button("Process")

## If ask button is clicked
if submit and uploaded_files is not None:
    combined_df = pd.DataFrame()
    
    for uploaded_file in uploaded_files:
        image = None
        if uploaded_file.type == "application/pdf":
            image = pdf_to_image(uploaded_file)
            #st.image(image, caption=f"Uploaded PDF Page as Image for {uploaded_file.name}.", use_column_width=(2,1))
        else:
            image = Image.open(uploaded_file)
            #st.image(image, caption=f"Uploaded Image for {uploaded_file.name}.", use_column_width=True)

        image_data = input_image_setup(uploaded_file)
        response_text = get_gemini_response(extraction_prompt, image_data, input_text)
        
        # Check if the response text is empty before proceeding
        if not response_text:
            st.warning(f"No valid response from the model for {uploaded_file.name}. Please check the pdf file.")
        else:
            #print(response_text)

            # Ensure that 'text' is a string
            if not isinstance(response_text, str):
                response_text = str(response_text)

            # Extract substrings
            substrings = extract_substrings(response_text)
            ##print(substrings)
            #st.subheader(f"Extracted substrings for {uploaded_file.name}:")
            #st.write(substrings)

            df = text_to_dataframe(substrings[0])
            #print(df)
            #st.subheader(f"Invoice Details Table for {uploaded_file.name}")
            #st.write(df)

            # Append the dataframe to the combined dataframe
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    st.subheader("Invoice Details CSV")
    st.write(combined_df)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'invoice_details_{current_time}.csv'
    csv = combined_df.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="Export to CSV",
    data=csv,
    file_name=file_name,
    mime='text/csv',)

