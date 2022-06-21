import streamlit as st
from bokeh.models.widgets import Div

import io
import os

import six
from urllib3.packages.six import MovedModule
from _io import TextIOWrapper

import webbrowser
from PIL import Image, ImageOps


from options.test_options import TestOptions
from models import create_model

from data.base_dataset import get_transform
from util import util

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
#opt.gpu_ids = str(-1)
opt.name = "latest_net_AE"
opt.load_size = 512+128

st.set_page_config(page_title='Image Style Transfer', layout="wide")

# from torch.nn.parameter import Parameter
# @st.cache(show_spinner=True, hash_funcs={Parameter: lambda _: None}, allow_output_mutation=True)
@st.cache(show_spinner=True, hash_funcs={six.MovedModule: lambda _: None, MovedModule: lambda _: None, TextIOWrapper: lambda _: None},
          allow_output_mutation=True)#, hash_funcs={Parameter: lambda _: None}, allow_output_mutation=True)
def load_model():
    model = create_model(opt)
    model.setup(opt)  # Loads the model weights
    model.eval()
    return model


def open_url(url):
    webbrowser.open_new_tab(url)


model = load_model()
transform = get_transform(opt)

st.image(os.path.join('Images', 'teaser.png'), use_column_width=True)#, width=1000)#, use_column_width=True)

st.title("Image Style Transfer")
st.subheader("Upload your own style & content images and watch your picture turn into a masterpiece!")
st.markdown("***")


m = st.markdown(
    """ <style> div.stButton > button:first-child { background-color: black; width: 18em; height: 3em; color: white} </style>""",
    unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: center; color: white;'>Check me out!</h1>", unsafe_allow_html=True)
if st.sidebar.button("Portfolio        "):
    js = "window.location.href = 'https://www.datascienceportfol.io/JohanBekker'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)
    # open_url('https://www.datascienceportfol.io/JohanBekker')
if st.sidebar.button("GitHub        "):
    #js = "window.open('https://github.com/JohanBekker')"  # New tab or window
    js = "window.location.href = 'https://github.com/JohanBekker'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)
    # open_url('https://github.com/JohanBekker')
if st.sidebar.button("LinkedIn        "):
    js = "window.location.href = 'https://www.linkedin.com/in/johan-bekker-3501a6168/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)
    # open_url('https://www.linkedin.com/in/johan-bekker-3501a6168/')

style = st.file_uploader('Upload your style image here', type=['jpg', 'jpeg', 'png'])
content = st.file_uploader('Upload your content image here', type=['jpg', 'jpeg', 'png'])

with st.expander("Configuration"):
    target_width = st.slider(
         'Select (shortest side) width to scale the image to. Because of Streamlit free deployment limitations, the image is'
         ' generated with a short side dimension of 640.',
         step=128, min_value=128, max_value=2048, value=640)
    st.markdown("***")
    grayscale = st.checkbox("Grayscale content image before style transfer")
    st.markdown("***")
    border = st.number_input('Add a border around your content image', min_value=0, max_value=500, value=0, step=10)

st.markdown("***")


col1, col2, col3 = st.columns(3)
if style and content:
    # image_A = Image.open(path_A).convert("RGB")
    # image_B = Image.open(path_B).convert("RGB")
    style = style.read()
    content = content.read()

    style = Image.open(io.BytesIO(style)).convert("RGB")
    content = Image.open(io.BytesIO(content)).convert("RGB")

    content = ImageOps.expand(content, border=int(border), fill="white")

    with col1:
        st.image(style)
    with col2:
        st.image(content)

    if grayscale:
        content = content.convert("L").convert("RGB")
        #content=


    A = transform(content).unsqueeze(0)
    B = transform(style).unsqueeze(0)
    dataset = {'A': A, 'B': B}

    model.set_input(dataset)
    model.test()

    visuals = model.get_current_visuals()
    im = util.tensor2im(visuals['fake_B'])
    image_pil = Image.fromarray(im)

    ow, oh = image_pil.size
    #target_width = 1024#512
    h = int(target_width * oh / ow)
    img2 = image_pil.resize((target_width, h), Image.BICUBIC)

    with col3:
        st.image(img2)

elif style or content:
    if style:
        with col1:
            st.image(style)
    if content:
        with col2:
            st.image(content)

st.markdown("***")
st.markdown("#### Based on [Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning (CAST)](https://github.com/zyxElsa/CAST_pytorch)")
st.write("Paper: [Arxiv](https://arxiv.org/abs/2205.09542)")

