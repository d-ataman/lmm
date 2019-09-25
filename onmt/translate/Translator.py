import torch
from torch.autograd import Variable
import torch.nn.functional as F
import onmt.translate.Beam
import onmt.io
from onmt.Models import RNNDecoderState
import numpy

class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, min_length=0):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        max_chars = 1
        batch_size = batch.batch_size
        src_data_type = data.src_data_type
        tgt_data_type = data.tgt_data_type
        vocab = self.fields["tgt"].vocab
        beam = [onmt.translate.Beam(tgt_data_type, max_chars,
                                    beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)


        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', src_data_type)
        src_lengths = None
        if src_data_type == 'words' and tgt_data_type == 'words':
            _, src_lengths = batch.src
            enc_states, memory_bank = self.model.encoder(src, src_lengths)
            dec_states = self.model.decoder.init_decoder_state(
                                        src, memory_bank, enc_states)
        elif src_data_type == "trigrams" and tgt_data_type == 'words':
            src_lengths = batch.src[1]
            enc_states, memory_bank = self.model.encoder(src, src_lengths, batch_size=batch_size)
            self.model.decoder.batch_size = batch_size
            dec_states = Variable(enc_states[0].data.new(enc_states[0].size()).zero_().unsqueeze(0).repeat(self.model.decoder.num_layers, 1, 2))
            dec_states = RNNDecoderState(self.model.decoder.hidden_size, dec_states)

        elif src_data_type == 'words' and tgt_data_type == 'characters':
            _, src_lengths = batch.src
            enc_states, memory_bank = self.model.encoder(src, src_lengths)
            self.model.decoder.batch_size = batch_size*beam_size

            dec_states = Variable(enc_states[0].data.new(enc_states[0].size()).zero_().unsqueeze(0).repeat(self.model.decoder.num_layers, 1, 2))
            dec_states = RNNDecoderState(self.model.decoder.hidden_size, dec_states)
        else:
            enc_states, memory_bank = self.model.encoder(src, src_lengths)
            dec_states = self.model.decoder.init_decoder_state(
                                        src, memory_bank, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))


        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if src_data_type in ['words', 'trigrams'] and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size) if src_data_type == 'words' else src_lengths[1].repeat(beam_size)
        
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.

            inp = var(torch.stack([b.get_current_state() for b in beam])
                       .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)
            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, attn, dec_states = self.model.decoder(
                inp, memory_bank, dec_states, memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
            else: # currently not supported for char-level decoding
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(
                    out[:, j],
                    unbottle(attn["std"]).data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        src_data_type = data.src_data_type
        if src_data_type == 'words':
            _, src_lengths = batch.src
        elif src_data_type == 'trigrams':
            src_lengths = batch.src[1]
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', src_data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        if src_data_type == "trigrams":
            enc_states, memory_bank = self.model.encoder(src, src_lengths, batch_size=batch.batch_size)
            self.model.decoder.batch_size = batch.batch_size
            src_lengths = src_lengths[1]
        else:
            enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, attn, dec_states = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores



    def greedy_translate(self, batch, data):
        """
        Translate a batch of sentences.
        Only used for character based decoding.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object

        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        max_chars = 50 #?
        self.max_length = 100
        batch_size = batch.batch_size
        src_data_type = data.src_data_type
        tgt_data_type = data.tgt_data_type
        vocab = self.fields["tgt"].vocab


        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', src_data_type)
        src_lengths = None

        _, src_lengths = batch.src

        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        self.model.decoder1.batch_size = batch_size
        self.model.decoder2.batch_size = batch_size
        word_state = self.model.decoder1.init_decoder_state(
                                        src, memory_bank, enc_states)


        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (3) run the decoder to generate sentences, using greedy search.

        # Initialize input of size 1 x batch_size x 1 (content: BOS)
        inp = torch.LongTensor(1, batch_size).fill_(2).cuda()
        inpp = inp.unsqueeze(2)


        outputs = []
        attns = []
        scores = []

        fin = numpy.zeros(batch_size)

        for i in range(self.max_length):

            if numpy.sum(fin) == batch_size:
                break

            # Run one step.
            attn, ctxs, word_state = self.model.decoder1(
                inpp, memory_bank, batch_size, word_state, memory_lengths=src_lengths)

            ctx = (ctxs[-1].unsqueeze(0),)
            char_state = self.model.decoder2.init_decoder_state(ctx, self.model.decoder2.embeddings.embedding_size)
            chars = []
            char_scores = []
            inp2 = torch.LongTensor(1, batch_size).fill_(5).cuda()
            inpp2 = inp2.unsqueeze(2)

            for c in range(max_chars):

                decoder_outputs, char_state = self.model.decoder2(inpp2, batch_size, ctxs, char_state, translate=True)

                # Take the output corresponding to the last predicted character
                dec_out = decoder_outputs[-1]
                out = dec_out.squeeze(1)

                # Predict the character
                char_probs = self.model.generator(out)
                char_probs = char_probs.permute(1,0)
                best_scores, best_scores_id = char_probs.topk(1,0,True,True)

                chars.append(best_scores_id.squeeze(0))
                char_scores.append(best_scores.squeeze(0))
                inpp2 = best_scores_id.unsqueeze(2)

            pred = torch.stack(chars, dim=0)
            score = torch.stack(char_scores, dim=0)
            output = torch.LongTensor(max_chars, batch_size).fill_(1).cuda()
            max_len = 0
            for j in range(batch_size):
                if pred[0,j] == 3:
                    output[0,j] = 3
                    fin[j] = 1
                    continue
                for k in range(max_chars):
                    if pred[k,j] == 4:
                        output[0:k+1,j] = pred[0:k+1,j]
                        if k > max_len:
                            max_len = k
                        break

            inpp = output[:max_len+1,:].unsqueeze(2)
            inpp = torch.cat([inp2.unsqueeze(2), inpp], dim=0)

            outputs += [output]
            attns += [attn['std'].expand(max_chars,batch_size,attn['std'].size(2))] # repeat attn weights for all the characters
            scores += [score]


        hyps = torch.cat(outputs, dim=0).unbind(1)
        scores = torch.cat(scores, dim=0).unbind(1)
        attns = torch.cat(attns, dim=0).unbind(1)


        #remove padding from output
        hyps_f = []
        scores_f = []
        attns_ff = []

        for k in range(batch_size):
            non_padding = hyps[k].ne(1)
            hyps_f.append(hyps[k].masked_select(non_padding))
            scores_f.append(scores[k].masked_select(non_padding))
            attns_f = []
            for atn in attns[k].unbind(1):
                attns_f.append(atn.masked_select(non_padding))
            attns_ff.append(torch.stack(attns_f, dim=1))


        ret = {"predictions": [],
               "scores": [],
               "attention": []}

        ret["predictions"] = hyps_f
        ret["attention"] = attns_ff
        ret["scores"] = scores_f
        ret["gold_score"] = [0] * batch_size
        ret["batch"] = batch

        return ret


    def beam_translate(self, batch, data):
        """
        Translate a batch of sentences.
        Only used for character based decoding.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object

        """

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        max_chars = 20
        self.max_length = 200
        batch_size = batch.batch_size
        src_data_type = data.src_data_type
        tgt_data_type = data.tgt_data_type
        vocab = self.fields["tgt"].vocab
        beam_size = self.beam_size

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', src_data_type)
        src_lengths = None

        _, src_lengths = batch.src

        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        self.model.decoder1.batch_size = batch_size
        self.model.decoder2.batch_size = batch_size
        word_state = self.model.decoder1.init_decoder_state(
                                        src, memory_bank, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (3) Run the decoder to generate sentences, using beam search

        # Initialize input of size 1 x batch_size x 1 (content: BOS)
        inp = torch.LongTensor(1, batch_size*beam_size).fill_(2).cuda()
        inpp = inp.unsqueeze(2)

        # repeat src variables and the word rnn state beam_size times
        word_state_hidden = word_state.hidden[0].split(1, dim=1)
        word_state_hidden = [x.repeat(1,beam_size,1) for x in word_state_hidden]
        word_state.hidden = (torch.cat(word_state_hidden, dim=1),)
        word_state.input_feed = word_state.input_feed.split(1, dim=1)
        word_state.input_feed = [x.repeat(1,beam_size,1) for x in word_state.input_feed]
        word_state.input_feed = torch.cat(word_state.input_feed, dim=1)

        memory_bank = memory_bank.data.split(1, dim=1)
        memory_bank = [x.repeat(1,beam_size,1) for x in memory_bank]
        memory_bank = torch.cat(memory_bank, dim=1)

        src_lengths = src_lengths.split(1, dim=0)
        src_lengths = [x.repeat(beam_size) for x in src_lengths]
        src_lengths = torch.cat(src_lengths, dim=0)

        # Output variables to be returned.
        outputs = []
        attns = []
        scores = []

        init_batch_size = batch_size #store original batch size
        beam_batch_size = init_batch_size*beam_size
        fin = numpy.zeros(init_batch_size) # decides when to quit decoding
        batch_size = beam_batch_size

        word_beam = [[] for m in range(beam_batch_size)]
        word_beam_scores = []
        word_hyps = [[] for m in range(init_batch_size)]
        word_hyps_scores = [[] for m in range(init_batch_size)]


        for i in range(self.max_length): # word-level loop

            if numpy.sum(fin) == init_batch_size:
                break

            # Run one step.
            attn, ctxs, word_state, _ = self.model.decoder1(
                inpp, memory_bank, beam_batch_size, word_state, memory_lengths=src_lengths, translate=True)

            ctx = (ctxs[-1].unsqueeze(0),)
            char_state = self.model.decoder2.init_decoder_state(ctx, self.model.decoder2.embeddings.embedding_size)
            inp2 = torch.LongTensor(1, beam_batch_size).fill_(5).cuda()
            inpp2 = inp2.unsqueeze(2)

            fin_c = numpy.zeros(beam_batch_size) #determines when to quit character beam search

            char_scores = []
            chars = []

            for char in range(max_chars): # char-level loop

                if numpy.sum(fin_c) == beam_batch_size:
                    break

                decoder_outputs, char_state = self.model.decoder2(inpp2, batch_size, ctxs, char_state, translate=True)

                # Take the output corresponding to the last predicted character
                dec_out = decoder_outputs[-1]
                out = dec_out.squeeze(1)

                # Predict the character
                char_probs = self.model.generator(out)
                char_probs = char_probs.permute(1,0)

                if char == 0:
                    best_scores, best_scores_id = char_probs.topk(beam_size,0,True,True)
                    next_chars = torch.cat(best_scores_id.unbind(1), dim=0)
                    next_char_scores = torch.cat(best_scores.unbind(1), dim=0)
                    # repeat character state beam_size times
                    char_state_hidden = char_state.hidden[0].split(1, dim=1)
                    char_state_hidden = [x.repeat(1,beam_size,1) for x in char_state_hidden]
                    char_state_hidden = torch.cat(char_state_hidden, dim=1)
                    char_state.hidden = (char_state_hidden,)
                    if i == 0:
                        batch_size = beam_size*batch_size
                    #init the beam
                    bl = next_chars.split(beam_size)
                    beam = [list(b.split(1)) for b in bl]
                    beam_scores = list(next_char_scores.split(beam_size))
                    c_hyps = [[] for m in range(beam_batch_size)]
                    c_hyps_scores=[[] for m in range(beam_batch_size)]
                else:
                    # iterate the character beam
                    best_scores, best_scores_id = char_probs.topk(beam_size,0,True,True)
                    possible_chars = best_scores_id.split(beam_batch_size, dim=1)
                    possible_char_scores = best_scores.split(beam_batch_size, dim=1)
                    next_chars= []
                    new_state = []
                    for k in range(beam_batch_size):
                        next_pr = beam_scores[k] + possible_char_scores[k]
                        new_leaves, indices = next_pr.view(-1).topk(beam_size,0,True,True)
                        next_beam = torch.gather(possible_chars[k].contiguous().view(-1), 0, indices)
                        temp_beam = [list(b.clone()) for b in beam[k]]
                        next_chars.append(next_beam)
                        temp_hidden = char_state.hidden[0][0,k*beam_batch_size:(k+1)*beam_batch_size,:].clone()
                        new_hidden = temp_hidden.clone()
                        # update the beam and the char hidden state
                        for num, idx in enumerate(indices % (beam_size)):
                            temp_beam[num] = torch.cat([beam[k][idx], next_beam[num].unsqueeze(0)])
                            new_hidden[num] = temp_hidden[idx].clone()
                        new_state.append(new_hidden)
                        beam[k] = [b.clone() for b in temp_beam]
                        beam_scores[k] = new_leaves.clone()
                        # dont let eow have children
                        for bi, b in enumerate(temp_beam):
                            if b[-1] == 4:
                                if len(c_hyps[k]) < batch_size:
                                    c_hyps[k].append(b)
                                    score = beam_scores[k][bi].clone()
                                    c_hyps_scores[k].append(score)
                                else:
                                    fin_c[k] = 1
                                beam_scores[k][bi] = -1000
                        # Pick the best hypotheses
                        if fin_c[k] == 1:
                            best_c_hyp_score, best_c_scores_id = torch.tensor(c_hyps_scores[k]).topk(beam_size,0,True,True)
                            best_c_hyp = [c_hyps[k][ind] for ind in best_c_scores_id]
                            c_hyps[k] = best_c_hyp
                            c_hyps_scores[k] = [best_c_hyp_score[ind] for ind in range(beam_size)]

                    next_chars = torch.cat(next_chars, dim=0)
                    char_state.hidden = (torch.cat(new_state, dim=0).unsqueeze(0),)

                # Feed the character RNN with the predicted character embeddings and forward decoding one more step
                next_chars = next_chars.unsqueeze(0)
                inpp2 = next_chars.unsqueeze(2)

            # Iterate the word beam
            maxlen = 0 # max word length
            new_state = []; new_input_feed = []

            for k in range(init_batch_size):
                w = []; w_scores = []

                if beam_size > 1:
                    if i == 0:
                        #init the word beam
                        best_hyp = c_hyps[k*beam_size]
                        best_hyp_score = torch.tensor(c_hyps_scores[k*beam_size])
                        word_beam[k*beam_size:(k+1)*beam_size] = best_hyp
                        word_beam_scores.append(best_hyp_score)
                    else:
                        w_hyps = c_hyps[k*beam_size:(k+1)*beam_size]
                        w_hyps_scores = c_hyps_scores[k*beam_size:(k+1)*beam_size]
                        for w_i in range(len(w_hyps)):
                            for w_j in range(len(w_hyps[w_i])):
                                w.append(w_hyps[w_i][w_j])
                                w_scores.append(w_hyps_scores[w_i][w_j].cuda())
                        prev_scores = word_beam_scores[k].split(1, dim=0)
                        prev_scores = [x.repeat(beam_size) for x in prev_scores]
                        prev_scores = torch.cat(prev_scores, dim=0).cuda()
                        p_w_scores = torch.stack(w_scores) + prev_scores
                        best_hyp_score, idx = p_w_scores.topk(beam_size,0,True,True)
                        best_hyp = [w[ind] for ind in idx] 
                        # update the beam and the word hidden state
                        temp_hidden = word_state.hidden[0][0,k*beam_size:(k+1)*beam_size,:].clone()
                        temp_if = word_state.input_feed[k*beam_size:(k+1)*beam_size,:].clone()
                        new_hidden = temp_hidden.clone()
                        new_if = temp_if.clone()
                        temp_beam = word_beam[k*beam_size:(k+1)*beam_size]
                        new_beam = [b.clone() for b in temp_beam]
                        for num, ind in enumerate(idx/beam_size):
                            new_beam[num] = torch.cat([temp_beam[ind], best_hyp[num]])
                            new_hidden[num] = temp_hidden[ind].clone()
                            new_if[num] = temp_if[ind].clone()
                        new_state.append(new_hidden)
                        new_input_feed.append(new_if)
                        word_beam[k*beam_size:(k+1)*beam_size] = [b.clone() for b in new_beam]
                        word_beam_scores[k] = best_hyp_score.clone()
                        for bi, b in enumerate(best_hyp): 
                            # dont let eos have children
                            if b[0] == 3:
                                #if len(word_hyps[k]) < batch_size:
                                word_hyps[k].append(new_beam[bi])
                                score = word_beam_scores[k][bi].clone()
                                word_hyps_scores[k].append(score)
                                word_beam_scores[k][bi] = -1000
                                #else:
                                #    fin[k] = 1
                                    # Pick the best hypotheses
                        

                else: # if beam_size=1
                    best_hyp_score = c_hyps_scores[k][0]
                    best_hyp = c_hyps[k][0].clone()
                    best_hyp = best_hyp.unsqueeze(0)
                    best_hyp_score = best_hyp_score.unsqueeze(0)
                    if best_hyp[0][0] == 3:
                        fin[k] = 1

                for l in range(beam_size):
                    if best_hyp[l].size(0) > maxlen:
                        maxlen = best_hyp[l].size(0)

                # Collect the output before being fed back to the word RNN
                chars.append(best_hyp)
                char_scores.append(best_hyp_score)

            if beam_size > 1 and i > 0:
                word_state.hidden = (torch.cat(new_state, dim=0).unsqueeze(0),)
                word_state.input_feed = torch.cat(new_input_feed, dim=0)

            pred = []
            for wo in chars:
                pred.append(torch.stack([F.pad(x, (0, maxlen - x.size(0)), 'constant', 1) for x in wo], dim=1))
            output = torch.cat(pred, dim=1)
            score = torch.cat(char_scores, dim=0)

            inp = output.unsqueeze(2)
            bow = torch.LongTensor(1, beam_batch_size).fill_(5).unsqueeze(2).cuda()
            inpp = torch.cat([bow, inp], dim=0)

            attns += [attn['std'].expand(maxlen,beam_batch_size,attn['std'].size(2))] # repeat attn weights for all the characters
            if beam_size == 1:
                outputs.append(output)
                scores.append(score)

        # Choose the best hypothesis for translation.
        if beam_size > 1:
            for k in range(init_batch_size):
                best_hyp_score, idx = torch.Tensor(word_hyps_scores[k]).topk(1,0,True,True)
                outputs.append(word_hyps[k][idx])
                scores.append(best_hyp_score)
        else:
            outputs = torch.cat(outputs, dim=0).unbind(1)
            scores = torch.stack(scores, dim=0).unbind(1)

        attns = torch.cat(attns, dim=0).unbind(1)
        #remove padding from output
        hyps_f = []
        scores_f = []
        attns_ff = []

        for k in range(init_batch_size):
            non_padding = outputs[k].ne(1)
            hyps_f.append(outputs[k].masked_select(non_padding))
            #scores_f.append(scores[k].masked_select(non_padding))
            scores_f.append(scores[k])
            attns_f = []
            for atn in attns[k].unbind(1):
                attns_f.append(atn)#.masked_select(non_padding))
            attns_ff.append(torch.stack(attns_f, dim=1))

        ret = {"predictions": [],
               "scores": [],
               "attention": []}

        ret["predictions"] = hyps_f
        ret["attention"] = attns_ff
        ret["scores"] = scores_f
        ret["gold_score"] = [0] * init_batch_size
        ret["batch"] = batch

        return ret


