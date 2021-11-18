# THISMOOSEDOESNOTEXIST
generate a (pretty poor) image of a moose, using pytorch

## Examples

<img src="https://i.imgur.com/zt73yIv.png" width="128"/> <img src="https://i.imgur.com/aALKmIC.png" width="128"/> <img src="https://i.imgur.com/zAF2ZIt.png" width="128"/> <img src="https://i.imgur.com/EsfYOik.png" width="128"/> <img src="https://i.imgur.com/71VH7CE.png" width="128"/>

## Setup
- Clone this repo
- Run the following commands:
    - `pip install torchvision`
    - `pip install flask`
    - `export FLASK_APP=server.py`
    - `flask run`
- Open your web browser of choice to http://127.0.0.1:5000/ (or whatever Flask says)
- Refresh the page to get new moose

## Why is it so bad?
1. I'm not really sure, my generator seems to fall far behind the discriminator each time I train
2. There isn't a good moose database, the images were *ripped* from Bing (and randomly manipulated with rotation and noise for variation)
3. This is my first pytorch program

## Why don't you host it?
1. I don't know how.
2. The site I used didn't give me enough free space to install pytorch.
