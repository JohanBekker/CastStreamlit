# Streamlit Deployment of Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning (CAST)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/johanbekker/caststreamlit/main/app.py)

I'm a big fan of vintage posters. In the past I have tried to make vintage posters out of pictures by scraping a ton of vintage posters from the web,
together with landscape pictures, and train a CycleGAN in the hopes that this would apply the style of vintage posters to the landscape pictures.
The results were a bit underwhelming, and in order to train the networks to generate images of decent quality a lot of compute was necessary. So I gave
up on my dreams and moved on with live.

Until I came across [Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning (CAST)](https://github.com/zyxElsa/CAST_pytorch)!
In this work, the authors tackle the challenging problem of arbitrary image style transfer using a novel style feature representation learning method.
A suitable style representation, as a key component in image stylization tasks, is essential to achieve satisfactory results.
Existing deep neural network based approaches achieve reasonable results with the guidance from second-order statistics such as Gram matrix of content features.
However, they do not leverage sufficient style information, which results in artifacts such as local distortions and style inconsistency.
To address these issues, the authors propose to learn style representation directly from image features instead of their second-order statistics, 
by analyzing the similarities and differences between multiple styles and considering the style distribution.

This repository is a fork of the original, with some alterations made for efficient inference. Unnecessary calculations have been commented out, and image scaling
has been fixed so that the input aspect ratio can be preserved. To show off this amazing work I've deployed my altered version as a Streamlit app. Play around with 
the app and turn your pictures into a Van Gogh masterpiece!

## Docker

To build this app in a Docker image, run the following bash command: 

```bash
docker build -t caststreamlit:latest -f docker/Dockerfile .
```

When the image is built, run it in a container:

```bash
docker run -p 8501:8501 caststreamlit:latest
```

Now you can reach your application in your webbrowser at http://localhost:8501/.

## Options

Because of Streamlit limitations, the model runs on cpu and the generated dimensions are capped to not exceed memory limits.
If you want to run it locally on gpu, navigate to options/base_options.py and change gpu_ids to 0.

To use the app to generate images of better quality, change the value of opt.load_size in app.py. To change the quality
without using the app, change the flag --load_size in options/base_options.py.

## Original usage (local)

### Datasets

   Then put your content images in ./datasets/{datasets_name}/testA, and style images in ./datasets/{datasets_name}/testB.
   
   Example directory hierarchy:
   ```sh
      CAST-pytorch
      |--- datasets
             |--- {datasets_name}
                   |--- testA
                   |--- testB
                   
      Then, call --dataroot ./datasets/{datasets_name}
   ```

### Test

   Test the CAST model:
   
   ```sh
   python test.py --dataroot ./datasets/{dataset_name} --name {model_name}
   ```
   
   The pretrained model is saved at ./checkpoints/CAST_model/*.pth.
   
   BaiduNetdisk: Check [here](https://pan.baidu.com/s/12oPk3195fntMEHdlsHNwkQ) (passwd：cast) 
   
   Google Drive: Check [here](https://drive.google.com/file/d/11dZqu95QfnAgkzgR1NTJfQutz8JlwRY8/view?usp=sharing)

## Citation
   
   ```sh
   @inproceedings{zhang2020cast,
   author = {Zhang, Yuxin and Tang, Fan and Dong, Weiming and Huang, Haibin and Ma, Chongyang and Lee, Tong-Yee and Xu, Changsheng},
   title = {Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning},
   booktitle = {ACM SIGGRAPH},
   year = {2022}}
   ```
   