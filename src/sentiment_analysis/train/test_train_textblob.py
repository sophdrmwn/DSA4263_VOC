import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '../src')

from train.train_textblob import *
import pytest

@pytest.fixture
def data(): 
    return ['This is a very healthy dog food. Good for their digestion. Also good for small puppies. My dog eats her required amount at every feeding.',
                                      "I've been very pleased with the Natural Balance dog food. Our dogs have had issues with other dog foods in the past and I had someone recommend Natural Balance grain free since it is possible they were allergic to grains. Since switching I haven't had any issues. It is also helpful that have have different kibble size for larger/smaller sized dogs.",
                                      "Before I was educated about feline nutrition, I allowed my cats to become addicted to dry cat food. I always offered both canned and dry, but wish I would have fed them premium quality canned food and limited dry food. I have two 15 year old cats and two 5 year old cats. The only good quality dry foods they will eat are Wellness and Innova. Innova's manufacturer was recently purchased by Procter&Gamble. I began looking for a replacement. After once again offering several samples (from my local holistic pet store) Holistic Select was the only one (other than the usual Wellness and Innova) they would eat. For finicky cats, I recommend trying Holistic Select. It is a good quality food that is very palatable for finicky eaters.",
                                      "I purchased this item because it was cheaper than other olive juices. i was looking for a good quality olive juice for martinis. i was pretty disappointed. the juice is very strong. i followed the instructions and put the ratio on the bottle into the shaker and made a martini. the ratio was 2 part vodka to 1 part olive juice. it was way too salty. the olive juice was over powering. i made a decent dirty martini with 2 parts vodka 1/4 part olive juice. i would not recommend this item.",
                                      "My husband just bought a Keurig coffe maker. In the box was samples, his favorite was Donut Shop. I ordered the 35 K-Cup for him. I was so disappointed, out of the 35 k-cups 11 were decaf, giving me 6 k-cups of each of the 4 flavors.<br />I also ordered the Wolfgang Puck sampler, I got 12 flavors 4 each. How cool is that!<br />I would order Donut Shop again, just not the sampler."]

def test_getSentiment():
    # check that getVaderSentiment returns 1 if value >=0
    value = 0
    assert getTextblobSentiment(value) == 1

def test_train_textblob_returns_correct_item(data):
    # check train_textblob returns correct items
    out = train_textblob(data)
    assert (len(out) == len(data)) & (type(out) == list)