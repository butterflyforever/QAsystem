import random

class Generator(object):
    def __init__(self):
        return
        
    def generate(self, candidates, query=None):
        if len(candidates):
            return candidates[random.randrange(len(candidates))]
        else:
            return None