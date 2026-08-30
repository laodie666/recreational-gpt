# Ref: https://github.com/ArslanJajja1/bigram-language-model/blob/main/bigram_scratch.py
import random

class BigramModelPython:
    def __init__(self):
        self.bigram_counts = {}
        self.bigram_probs = {}
        self.vocab = set()

    def train(self, text:str):


        self.vocab = self.vocab.union(set(text))

        for i in range(len(text)-1):
            cur_char = text[i]
            next_char = text[i+1]
            if cur_char not in self.bigram_counts:
                self.bigram_counts[cur_char] = {} 
            if next_char not in self.bigram_counts[cur_char]:
                self.bigram_counts[cur_char][next_char] = 1
            else:
                self.bigram_counts[cur_char][next_char] += 1


        # print("vocab", self.vocab)
        # print("bigram count", self.bigram_counts)


        for cur_char, next_chars in self.bigram_counts.items():
            sum = 0
            for next_char in next_chars:
                sum += self.bigram_counts[cur_char][next_char]

            for next_char in next_chars:
                if cur_char not in self.bigram_probs:
                    self.bigram_probs[cur_char] = {} 
                self.bigram_probs[cur_char][next_char] = self.bigram_counts[cur_char][next_char]/sum

        # print("bigram probs", self.bigram_probs)
        print("training done")

    def generate(self, start_char, length = 100) -> str:
        
        txt = start_char
        cur_char = start_char
        for i in range(length):
            if cur_char not in self.bigram_probs:
                cur_char = random.choice(list(self.vocab))
                txt += cur_char
                continue

            keys = list(self.bigram_probs[cur_char].keys())
            values = list(self.bigram_probs[cur_char].values())
            next_char = random.choices(keys, values)[0]
            txt += next_char
            cur_char = next_char
        return txt


if __name__ == "__main__":
    data = open("input.txt").read()
    model = BigramModelPython()
    model.train(data)
    print(model.generate("h"))