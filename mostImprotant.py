import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.utils import get_stop_words



text =         """Nvidia has announced a new chip for PCs as it moves into the consumer market for devices integrated with AI technology. "This reinvention of the computer is as big of a deal as the reinvention of the phone into what we now know as the smartphone," Nvidia's chief executive Jensen Huang said as he unveiled the RTX Spark chip. Huang made the announcement on Monday as he delivered a keynote speech ahead of the opening of the Computex technology show in Taipei, Taiwan. Separately on Sunday, the US tightened its rules on selling Nvidia's most advanced chips to Chinese firms. The RTX Spark is "a new superchip... for the era of personal AI agents - offering a new class of computer that moves from tool to teammate," Nvidia said on its website. It will be included in a new line of Windows PCs made by Lenovo, HP, Dell, Microsoft Surface, Asus, and MSI. They are due to be available in the autumn, with models from Acer and Gigabyte to follow. The move marks a challenge to high-profile names in the PC market like Apple and Intel. Lenovo, HP, Dell and Apple accounted for almost 75% of the world 's PC market in the first three months of this year, according to research firm Gartner."""


parser = PlaintextParser.from_string(text, Tokenizer("english"))

summarizer = LexRankSummarizer()
summarizer.stop_words = get_stop_words("english")

summary = summarizer(parser.document, 3)   

print("The most important sentences are:")
for sentence in summary:
    print(sentence)