from prepare_pop import get_train_data, get_pop
from controller import train_seq2seq
import d2l
from evolutionary_algorithm import Evolutionary

class AutoFE:
    def __init__(self):
        self.evolutionary = Evolutionary()

    def train(self, args):
        train_data = get_train_data()
        init_pops = get_pop()
        d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
        engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
        fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
        for eng, fra in zip(engs, fras):
            translation, dec_attention_weight_seq = d2l.predict_seq2seq(
                net, eng, src_vocab, tgt_vocab, num_steps, device, True)
            print(f'{eng} => {translation}, ',
                  f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

        # TODO: 应该增加长度惩罚、拓展采样(Top-k Sampling,Top-p Sampling)
        for pop in init_pops:
            embedding_pop = Encoder(pop)


        final_embedding_pops = self.evolutionary.train(embedding_pops)



        for embedding_pops in final_embedding_pops:
            translated_list = ['SEP','EOS']
            for token in
            translation, dec_attention_weight_seq = d2l.predict_seq2seq(
                net, token, src_vocab, tgt_vocab, num_steps, device, True)
            translated_list.append(translation)
