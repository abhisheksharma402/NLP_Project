import streamlit as st
import re
from bs4 import BeautifulSoup
import nltk
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset

st.set_page_config(
    page_icon="📊",  # Set page icon
    layout="wide"  # Set wide layout
)


def predict_class(model_path, input_text):

    # Load BertForSequenceClassification 
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels = 2,   # The number of output labels--2 for binary classification.
        output_attentions = False, 
        output_hidden_states = False, 
    )


    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # model = torch.load(model_path, map_location=torch.device('cpu'))
    # print(model)
    # model.invoke()
    # model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode

    test_input_ids = []
    test_attention_masks = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text_encoding = tokenizer(input_text, truncation = True, padding = True)
    
    encoded_dict = tokenizer.encode_plus(
                        input_text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256,          # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )

    # Add the encoded sentence to the list.
    test_input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    test_attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    test_input_ids = torch.cat(test_input_ids, dim=0)
    test_attention_masks = torch.cat(test_attention_masks, dim=0)
    # test_labels = torch.tensor(test_labels)

    # # Set the batch size.
    # batch_size = 32

    # # Create the DataLoader.
    # prediction_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    # prediction_sampler = SequentialSampler(prediction_data)
    # prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        result = model(test_input_ids,
                        token_type_ids=None,
                        attention_mask=test_attention_masks,
                        return_dict=True)
            
    logits = result.logits
    predicted_class = torch.argmax(logits).item()


    return predicted_class




def remove_bullet_points(text):
    # Define pattern to match bullet points
    bullet_point_pattern = re.compile(r'\s*[\u2022\u2023\u25E6]\s*')  # Matches •, ‣, and ◦ bullet points

    # Remove bullet points from the text
    cleaned_text = bullet_point_pattern.sub(' ', text)
    
    return cleaned_text

def remove_emojis_and_symbols_from_text(text):
    # Define pattern to match emojis
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           u"\u00ae" # trade Marks ®
                           u"\u00A9" # copy Right ©
                           u"\u2122" # Trade Mark TM
                           u"\u200b"
                           "]+", flags=re.UNICODE)
    # Remove emojis from the text
    cleaned_text = emoji_pattern.sub(r' ', text)
    return cleaned_text

def decontraction(x):
    contractions = {
        "you've": 'you have',
        "you're": 'you are',
        "haven't": 'have not',
        "hasn't": 'has not',
        "hadn't've": 'had not have',
        "hadn't": 'had not',
        "don't": 'do not',
        "doesn't": 'does not',
        "didn't": 'did not',
        "couldn't've": 'could not have',
        "couldn't": 'could not',
        "could've": 'could have',
        "'cause": 'because',
        "can't've": 'cannot have',
        "aren't": 'are not',
        "ain't": 'am not',
        "can't": 'can not',
        "won't": 'will not',
        "he'd": 'he would',
        "he'd've": 'he would have',
        "he'll": 'he will',
        "he'll've": 'he will have',
        "he's": 'he is',
        "how'd": 'how did',
        "how'd'y": 'how do you',
        "how'll": 'how will',
        "how's": 'how does',
        "i'd": 'i would',
        "i'd've": 'i would have',
        "i'll": 'i will',
        "i'll've": 'i will have',
        "i'm": 'i am',
        "i've": 'i have',
        "isn't": 'is not',
        "it'd": 'it would',
        "it'd've": 'it would have',
        "it'll": 'it will',
        "it'll've": 'it will have',
        "it's": 'it is',
        "let's": 'let us',
        "ma'am": 'madam',
        "mayn't": 'may not',
        "might've": 'might have',
        "mightn't": 'might not',
        "mightn't've": 'might not have',
        "must've": 'must have',
        "mustn't": 'must not',
        "mustn't've": 'must not have',
        "needn't": 'need not',
        "needn't've": 'need not have',
        "o'clock": 'of the clock',
        "oughtn't": 'ought not',
        "oughtn't've": 'ought not have',
        "shan't": 'shall not',
        "sha'n't": 'shall not',
        "shan't've": 'shall not have',
        "she'd": 'she would',
        "she'd've": 'she would have',
        "she'll": 'she will',
        "she'll've": 'she will have',
        "she's": 'she is',
        "should've": 'should have',
        "shouldn't": 'should not',
        "shouldn't've": 'should not have',
        "so've": 'so have',
        "so's": 'so is',
        "that'd": 'that would',
        "that'd've": 'that would have',
        "that's": 'that is',
        "there'd": 'there would',
        "there'd've": 'there would have',
        "there's": 'there is',
        "they'd": 'they would',
        "they'd've": 'they would have',
        "they'll": 'they will',
        "they'll've": 'they will have',
        "they're": 'they are',
        "they've": 'they have',
        "to've": 'to have',
        "wasn't": 'was not',
        "we're": 'we are',
        "'s": " is",
        "n't": ' not',
        "'re": ' are',
        "'d": ' would',
        "'ll": ' will',
        "'t": ' not',
        "'ve": ' have',
        "'m": ' am',
        ' u ': ' you ',
        ' ur ': ' your ',
        ' n ': ' and '
    }
    
    contractions_comma = {
        "you’ve": 'you have',
        "you’re": 'you are',
        "haven’t": 'have not',
        "hasn’t": 'has not',
        "hadn’t've": 'had not have',
        "hadn’t": 'had not',
        "don’t": 'do not',
        "doesn’t": 'does not',
        "didn’t": 'did not',
        "couldn’t’ve": 'could not have',
        "couldn’t": 'could not',
        "could’ve": 'could have',
        "’cause": 'because',
        "can’t’ve": 'cannot have',
        "aren’t": 'are not',
        "ain’t": 'am not',
        "can’t": 'can not',
        "won’t": 'will not',
        "he’d": 'he would',
        "he'd've": 'he would have',
        "he’ll": 'he will',
        "he’ll’ve": 'he will have',
        "he’s": 'he is',
        "how’d": 'how did',
        "how’d'y": 'how do you',
        "how’ll": 'how will',
        "how’s": 'how does',
        "i’d": 'i would',
        "i’d’ve": 'i would have',
        "i’ll": 'i will',
        "i’ll’ve": 'i will have',
        "i’m": 'i am',
        "i’ve": 'i have',
        "isn’t": 'is not',
        "it’d": 'it would',
        "it’d’ve": 'it would have',
        "it’ll": 'it will',
        "it'll've": 'it will have',
        "it’s": 'it is',
        "let's": 'let us',
        "ma’am": 'madam',
        "mayn’t": 'may not',
        "might’ve": 'might have',
        "mightn’t": 'might not',
        "mightn’t've": 'might not have',
        "must’ve": 'must have',
        "mustn’t": 'must not',
        "mustn’t’ve": 'must not have',
        "needn’t": 'need not',
        "needn’t’ve": 'need not have',
        "o’clock": 'of the clock',
        "oughtn’t": 'ought not',
        "oughtn’t’ve": 'ought not have',
        "shan’t": 'shall not',
        "sha’n’t": 'shall not',
        "shan’t’ve": 'shall not have',
        "she’d": 'she would',
        "she’d’ve": 'she would have',
        "she’ll": 'she will',
        "she’ll’ve": 'she will have',
        "she’s": 'she is',
        "should’ve": 'should have',
        "shouldn’t": 'should not',
        "shouldn't've": 'should not have',
        "so’ve": 'so have',
        "so's": 'so is',
        "that’d": 'that would',
        "that’d've": 'that would have',
        "that’s": 'that is',
        "there’d": 'there would',
        "there’d’ve": 'there would have',
        "there’s": 'there is',
        "they’d": 'they would',
        "they’d've": 'they would have',
        "they’ll": 'they will',
        "they’ll've": 'they will have',
        "they’re": 'they are',
        "they’ve": 'they have',
        "to’ve": 'to have',
        "wasn’t": 'was not',
        "we’re": 'we are',
        "’s": " is",
        "n’t": ' not',
        "’re": ' are',
        "’d": ' would',
        "’ll": ' will',
        "’t": ' not',
        "’ve": ' have',
        "’m": ' am',
        ' u ': ' you ',
        ' ur ': ' your ',
        ' n ': ' and '
    }
    
    for key in contractions:
        value = contractions[key]
        x = x.replace(key, value)
        
    for key in contractions_comma:
        value = contractions_comma[key]
        x = x.replace(key, value)
    return x

def removepunc(x):
    punctuation = '!\xad\xa0\xe2\x80\x9d\xe2\x80\x99\xe2\x80\xa2()*+-/:;<=>[]^_`{|}~@#,.?$%&"”“’‘\'…<>«»'
    translation_table = str.maketrans(punctuation, ' ' * len(punctuation))
    return x.translate(translation_table)

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text)

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)

def remove_ampersand(text):
    text = text.replace('&amp;', '')
    text = text.replace('andamp;', '')
    return text


def remove_code_from_text(text):
    # Defining the pattern to match the code
    code_pattern = r'\b[0-9a-fA-F]{64}\b'  # Assuming the code is a 64-character hexadecimal string

    # Remove the code from the text
    cleaned_text = re.sub(code_pattern, ' ', text)

    return cleaned_text

def remove_number_from_text(text):
    # Defining the pattern to match the code
    number_pattern = r'[0-9]'  # Assuming the code is a 64-character hexadecimal string

    # Remove the code from the text
    cleaned_text = re.sub(number_pattern, ' ', text)

    return cleaned_text

def remove_greek_words(text):
    # Define pattern to match Greek words
    greek_pattern = re.compile(r'\b[α-ωΑ-Ωίϊΐόάέύϋΰήώ]+\b', flags=re.IGNORECASE)

    # Remove Greek words from the text
    cleaned_text = greek_pattern.sub(' ', text)
    
    return cleaned_text

def remove_russian_words(text):
    # Define pattern to match Russian words
    russian_pattern = re.compile(r'\b[а-яА-Я]+\b')  # Matches Russian words

    # Remove Russian words from the text
    cleaned_text = russian_pattern.sub('', text)
    
    return cleaned_text

def remove_thai_words(text):
    # Remove Thai characters using regular expression
    cleaned_text = re.sub(r'[\u0E00-\u0E7F]', ' ', text)
    
    return cleaned_text


def remove_currency_symbols(text):
    # Define pattern to match currency symbols
    currency_pattern = re.compile(r'[$€¥₹£¢₽₩₪₴₱₨฿₦₮₲₭₵₿]')  # Matches any currency symbol

    # Remove currency symbols from the text
    cleaned_text = currency_pattern.sub(' ', text)
    
    return cleaned_text


def remove_hyphens(text):
    # Define pattern to match hyphens within words
    hyphen_pattern = re.compile(r'\b(\w+)[-—‑–](\w+)\b')  # Matches hyphens within words

    # Remove hyphens from the text
    cleaned_text = hyphen_pattern.sub(r'\1 \2', text)
    
    return cleaned_text


def text_processing(text):
    
    # Lowercasing text
    text = text.lower()
    
    # Removing URLs from text
    text = re.sub(r"http\S+", "", text)
    
    # Removing newline character
    text = re.sub(r"\n", "", text)
    
    # Replacing & character with 'and'
    text = re.sub(r"&", "and", text)
    
    # Removing HTML tags
    text = remove_html_tags(text)
    
    # Removing ampersand from the text 
    text = remove_ampersand(text)
    
    # Performing decontraction on the text
    text = decontraction(text)
    
    # Removing the punctuations from the text 
    text = removepunc(text)
    
    # Removing 64-character hexadecimal string in the text
    text = remove_code_from_text(text)
    
    # Removing number in the text
    text = remove_number_from_text(text)
    
    # Removing emojis from text
    text = remove_emojis_and_symbols_from_text(text)  
    
    # Remove •, ‣, and ◦ bullet points
    text = remove_bullet_points(text)
    
    # Remove greek words from the text
    text = remove_greek_words(text)
    
    # Remove russian words from the text
    text = remove_russian_words(text)
    
    # Remove thai words from the text
    text = remove_thai_words(text)
    
    # Remove hyphens from hyphenated word
    text = remove_hyphens(text)
    
    # Remove currency symbols from the text
    text = remove_currency_symbols(text)
    
    # Removing extra space in the text
    text = remove_extra_spaces(text)
    
    return text





# Define function to apply custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load local CSS file
local_css("style.css")

# Define a function to create visually appealing header
def header():
    st.markdown("<h1 style='text-align: center; color: #2A9DF4;'>Fake Job Recruitment Post Detector</h1>", unsafe_allow_html=True)



# Create an expandable container for the header
with st.expander("", expanded=True):
    header()

st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Enter Job Post Details Below:</h3>", unsafe_allow_html=True)
# Input elements
st.markdown("<h2 style='color: #333;'>Job Post Details</h2>", unsafe_allow_html=True)

# Use CSS to display inputs in the same line
st.write('<div class="input-container">', unsafe_allow_html=True)
title = st.text_input("Title")
company_profile = st.text_area("Company Profile")
st.write('</div>', unsafe_allow_html=True)

description = st.text_area("Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")


final_text = title +" "+company_profile+" "+description+" "+description+" "+requirements+" "+benefits

preprocessed_final_text = text_processing(final_text)

print(preprocessed_final_text)

# Button to make prediction
st.markdown("<br>", unsafe_allow_html=True)  # Add some space

model_path = "best_model.pth"

if st.button("Make Prediction"):
    # Your prediction logic goes here
    st.write("Prediction result will be displayed here...")

    # predicted_class = predict(preprocessed_final_text)

    output = predict_class(model_path, preprocessed_final_text)



    # print(predicted_class)
    # print(get_prediction(model, preprocessed_final_text))

    if(output == 0):
        st.write("<p id='output_g'>This is highly likely to be a <b>Genuine</b> Job Post</p>", unsafe_allow_html=True)
    else:
        st.write("<p id='output_f'>This is highly likely to be a <b>Fraudulent</b> Job Post</p>", unsafe_allow_html=True)
    

    # st.write(output)
