# a script to download free to use images with given keywords from google

from simple_image_download import simple_image_download as simp

downloader = simp.simple_image_download()

#keywords = ["lane stripes"]
keywords = ["traffic light"]
for kw in keywords:
    # simple_images get created in current folder, change folder have to be taken manually
    downloader.download(kw, 20)