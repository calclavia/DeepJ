from tqdm import tqdm
from midi_io import *
import constants as const
import sentencepiece as spm

def build_vocab(vocab_size=const.VOCAB_SIZE, chunk_size=2 ** 13):
    print('Dumpling MIDI samples...')
    with open('/tmp/token.txt', 'w') as f:
        for fname in tqdm(get_all_files(const.STYLES)):
            seq = load_midi(fname)
            token_str = tokens_to_str(seq)
            chunks = [token_str[i:i+chunk_size] + '\n' for i in range(0, len(token_str), chunk_size)]
            f.writelines(chunks)

    spm.SentencePieceTrainer.Train('--input=/tmp/token.txt --model_prefix=out/token --vocab_size={} --model_type=bpe'.format(vocab_size))

if __name__ == '__main__':
    build_vocab()
