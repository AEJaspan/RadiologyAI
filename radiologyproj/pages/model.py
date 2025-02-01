import torch
import streamlit as st
from radiologyproj.model.utils import generate_caption, lead_and_transform, tokenize_prompt, tokenizer
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


# Display images
def display_images(images):
    display_images = torch.stack(images, dim=0)
    plt.imshow(make_grid(display_images, normalize=True).permute(1, 2, 0))
    st.pyplot(plt)


# Streamlit UI
st.title("CXRMate Image Caption Generator")
st.write("Upload medical images, and generate captions describing the findings.")

# # File upload widget for images
# uploaded_image_1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
# uploaded_image_2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])
uploaded_files = st.file_uploader("Choose files", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
if len(uploaded_files) > 2:
    st.warning(
        "Warning: You can only upload two images at a time."
        "Only the first two images will be processed."
    )
    uploaded_files = uploaded_files[:2]
if uploaded_files:
    previous_findings = [None, None]
    previous_impression = [None, None]
    prompt = tokenize_prompt(previous_findings, previous_impression, tokenizer, 256, add_bos_token_id=True)
    images_transformed = [lead_and_transform(image_path) for image_path in uploaded_files]
    display_images(images_transformed)
    images = torch.stack(images_transformed, dim=0)
    ### NOTE - I need to figure out what images should be passed here ###
    ### TODO - Remove [images]*2 ###
    images = torch.nn.utils.rnn.pad_sequence([images]*2, batch_first=True, padding_value=0.0)
    with st.spinner("Generating caption..."):
        caption = generate_caption(images, prompt)
    
    # Show the generated caption
    st.subheader("Generated Caption:")
    st.write(caption)
else:
    st.warning("Please upload two images for caption generation.")
