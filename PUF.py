import random
import numpy as np
import math
import itertools

class PUF:

    def __init__(self, bits, w=[]):
        self.bits = bits
        self.w = w if len(w) != 0 else [ random.uniform(-1,1) for _ in range(bits) ]
        
    def calculate_responses(self):
        response = []
        
        for seq in itertools.product("01", repeat=self.bits):
            challenge = [int(item) for item in seq]
            response.append(self.calculate_one(challenge))

        return response

    def calculate_responses_with_random_challenges(self, nr):
        response = []
        
        for _ in range(nr):
            chal = [ random.randint(0, 1) for _ in range(self.bits) ]
            response.append(self.calculate_one(chal))
        
        return response

    def calculate_one(self, challenge=[]):
        if len(challenge) == 0:
            challenge = [ random.randint(0, 1) for _ in range(self.bits) ]
        
        response = challenge.copy()
        
        phi = [self._calculate_phi(challenge[i:]) for i in range(len(challenge))]
          
        delay = np.dot(np.transpose(self.w), phi)

        response.append(0 if delay < 0 else 1)
        return response

    def _calculate_phi(self, challenges):
        return np.prod([math.pow(-1, c) for c in challenges])