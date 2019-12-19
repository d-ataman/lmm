from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.Samplers import DiagonalGaussianSampler, KumaSampler

import numpy as np

import onmt
from onmt.Utils import aeq
#from onmt.modules.Samplers import DiagonalGaussianSampler


def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None, encoder_state=None):
        """
        Args:
            src (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            encoder_state (rnn-class specific):
               initial encoder_state state.

        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = (mean, mean)
        return encoder_final, memory_bank


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for i in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, dropout=dropout
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        assert isinstance(state, RNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        # END

        # Run the forward pass of the RNN.
        decoder_outputs, attns, decoder_final = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return decoder_outputs, attns, state

    def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        decoder_outputs, p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                decoder_outputs.view(-1, decoder_outputs.size(2))
            )
            decoder_outputs = \
                decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)

        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_outputs, attns, decoder_final

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Memory_Bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn = self.attn(
                rnn_output, memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return decoder_outputs, attns, hidden

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, attns, dec_state = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate=None):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        batch_size = self.hidden[0].size(1)
        
        self.coverage = None

        # Init the input feed.
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(self.hidden[0].data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage): 
    # rnnstate is a list of num_layers elements where each state corresponds to one layer

        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

class RNNTrigramsEncoder(RNNEncoder):

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNTrigramsEncoder, self).__init__\
            (rnn_type, bidirectional, num_layers,
             hidden_size, dropout, embeddings,
             use_bridge)
        self.rnn_trigram, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=embeddings.embedding_size//2,
                        num_layers=1,
                        dropout=dropout,
                        bidirectional=bidirectional)
        self.combine_states = nn.Linear(embeddings.embedding_size, embeddings.embedding_size)
        self.rnn_size = embeddings.embedding_size
        self.dropout = nn.Dropout(dropout)

    def _check_args(self, input, lengths=None, hidden=None):
        n_chars, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths[0].size()
            aeq(n_batch, n_batch_)

    def forward(self, src, batch_size, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        trigrams_len, batch_seqlen, emb_dim = emb.size()
        packed_emb = emb
        if lengths is not None: #and not self.no_pack_padded_seq:
        #    # Lengths data is wrapped inside a Variable.
            lengs = lengths[0].view(-1).data.tolist()
            seq_lens = lengths[1].view(-1).tolist()
        #    packed_emb = pack(emb, lengths)

        words_memory_bank, trigram_final = self.rnn_trigram(packed_emb)
        back_state = words_memory_bank[0, :, self.rnn_size//2:]
        forw_state = []
        #for i in range(words_memory_bank.size(1)):
        #    forw_state += [words_memory_bank[lengs[i]-1, i, :self.rnn_size//2].view(1, self.rnn_size//2)]
        forw_state = words_memory_bank[-1, :, :self.rnn_size//2]
        #forw_state = torch.cat(forw_state)
        word_embs = self.combine_states(torch.cat([back_state, forw_state], dim=-1))
        word_embs = self.dropout(word_embs)
        word_embs = torch.cat(word_embs.unsqueeze(1).split(batch_seqlen // batch_size, dim=0), dim=1)
        word_embs = pack(word_embs, seq_lens)
        memory_bank, encoder_final = self.rnn(word_embs, encoder_state)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        if self.use_bridge:
            encoder_final = self._bridge(self.dropout(encoder_final))
        return encoder_final, memory_bank


class RNNWordDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class for word decoding.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.TargetCharNMTModel`.
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNWordDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, batch_size, state, memory_lengths=None, translate=False):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        assert isinstance(state, RNNDecoderState)
        # END

        # Run the forward pass of the RNN.
        attns, outputs, hidden, ctx, logloss = self._run_forward_pass(
            tgt, memory_bank, batch_size, state, memory_lengths=memory_lengths, translate=False)

        # Update the state with the result.
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(hidden, ctx, coverage)


        for k in attns:
            attns[k] = torch.stack(attns[k])

        return attns, outputs, state, logloss

    def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))



class RNNCharDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class for character decoding.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.TargetCharNMTModel`.


    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNCharDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.char_decoder = self._build_rnn(rnn_type,
                                   input_size=self.embeddings.embedding_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   dropout=dropout)


    def forward(self, tgt, batch_size, hiddens, char_state, translate=False):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the char decoder
        """

        # Check
        assert isinstance(char_state, RNNDecoderState)
        # END

        # Run the forward pass of the RNN.
        decoder_outputs, char_hidden = self._run_forward_pass(
            tgt, batch_size, hiddens, char_state, translate)

        # Update the state with the result.
        char_state.hidden = char_hidden

        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)

        return decoder_outputs, char_state

    def init_decoder_state(self, hidden, embedding_size):

        #char_hidden = self.tanh(self.word2char(hidden))
        char_hidden = hidden
        if isinstance(char_hidden[0], tuple):  # LSTM
            return RNNDecoderState(embedding_size, tuple([char_hidden]))
        else:  # GRU
            return RNNDecoderState(embedding_size, char_hidden)



class StdCharRNNDecoder(RNNCharDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(StdCharRNNDecoder, self)\
            .__init__(rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn,
                 context_gate, copy_attn, dropout, embeddings,
                 reuse_copy_attn)


    def _run_forward_pass(self, tgt, batch_size, hiddens, char_state, translate=False):

        # Initialize local and return variables.
        decoder_outputs = []

        chars_len, batch_seqlen, _ = tgt.size()
        seqlen = batch_seqlen // batch_size
 
        char_embs = self.embeddings(tgt)

        for i, hidden_w in enumerate(hiddens):
            # Initialize the char LSTM input feed and the state at the beginning of each word.
            if translate == False:
                char_s = (hidden_w.unsqueeze(0),)
            else:
                char_s = char_state.hidden

            for j in range(chars_len):
                charrnn_input = char_embs[j,i*batch_size:(i+1)*batch_size,:]
                charrnn_output, char_s = self.char_decoder(charrnn_input, char_s)
                charrnn_output = charrnn_output + charrnn_input
                charrnn_output = self.dropout(charrnn_output)
                decoder_outputs += [charrnn_output]

        return decoder_outputs, char_s

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size



class StdWordRNNDecoder(RNNWordDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(StdWordRNNDecoder, self)\
            .__init__(rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn,
                 context_gate, copy_attn, dropout, embeddings,
                 reuse_copy_attn)
        self.char_rnn, _ = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=embeddings.embedding_size//2,
                        num_layers=1,
                        dropout=dropout,
                        bidirectional=True)
        self.combine_states = nn.Linear(embeddings.embedding_size, embeddings.embedding_size)
        self.rnn_size = embeddings.embedding_size
        self.dropout = nn.Dropout(p=dropout)
        self.word2char = nn.Linear(self.rnn_size*3, self.rnn_size, bias=True)
        self.tanh = nn.Tanh()
        self.lemma_dim = 100
        self.kuma_dim = 10
        self.sampler_z = DiagonalGaussianSampler(self.lemma_dim, hidden_size)
        self.substract = nn.Linear(self.lemma_dim, hidden_size, bias=True)
        self.sampler_f = KumaSampler(self.kuma_dim, hidden_size+self.lemma_dim)
        self.wordcomposition = nn.Linear(self.lemma_dim+self.kuma_dim, hidden_size, bias=True)

    def _run_forward_pass(self, tgt, memory_bank, batch_size, state, memory_lengths=None, translate=False):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """

        # Additional args check.
        input_feed = state.input_feed #.squeeze(0)

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        # Predict word representations with the bi-directional char-rnn
        char_embs = self.embeddings(tgt)
        assert char_embs.dim() == 3  # len x batch x embedding_dim
        chars_len, batch_seqlen, emb_dim = char_embs.size()
        
        words_memory_bank, final_char = self.char_rnn(char_embs)
        words_memory_bank = words_memory_bank + char_embs # resid connection

        back_state = words_memory_bank[0, :, self.rnn_size//2:]
        forw_state = []
        forw_state = words_memory_bank[-1, :, :self.rnn_size//2]
        word_embs = self.combine_states(torch.cat([back_state, forw_state], dim=-1))
        word_embs = torch.cat(word_embs.unsqueeze(1).split(batch_seqlen // batch_size, dim=0), dim=1) # n_words x batch_size x dim

        word_embs = self.dropout(word_embs)

        tgt_len, tgt_batch, _ = word_embs.size()

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(word_embs.split(1)):

            if emb_t.dim() > 2:
                emb_t = emb_t.squeeze(0)
            if input_feed.dim() > 2:
                input_feed = input_feed.squeeze(0)

            # Get the word prediction with the word-level RNN.
            wordrnn_input = torch.cat([emb_t, input_feed], 1)

            #import pdb; pdb.set_trace()
            wordrnn_output, hidden = self.rnn(wordrnn_input, hidden)

            wordrnn_output = wordrnn_output + emb_t # resid connection
            wordrnn_output = self.dropout(wordrnn_output)

            # Sample the latent variables
            z = self.sampler_z(wordrnn_output, batch_size, translate) # lemma
            f, logloss = self.sampler_f(torch.cat([wordrnn_output, z], dim=1), batch_size, translate) # morphological features
            #word_rep = self.tanh(self.wordcomposition(torch.cat([z, f], dim=1)))
            word_rep = self.wordcomposition(torch.cat([z, f], dim=1))

            # Get the predicted word using the attention.
            attn_out, p_attn, ctx = self.attn(
                word_rep,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                word_pred = self.context_gate(
                    wordrnn_input, wordrnn_output, word_pred
                )

            ctx = self.dropout(ctx)
            input_feed = ctx

            outputs += [attn_out]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(word_pred,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]

            # Return result.
        return attns, outputs, hidden, ctx, logloss

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size

class NMTSourceTrigramModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTSourceTrigramModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.bidirectional_encoder = True

    def forward(self, src, tgt, lengths, batch_size, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_final, memory_bank = self.encoder(src, batch_size, lengths)
        enc_state = Variable(enc_final[0].data.new(enc_final[0].size()).zero_().unsqueeze(0).repeat(self.decoder.num_layers, 1, 2))
        enc_state = RNNDecoderState(self.decoder.hidden_size, enc_state)
            #self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths[1])
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state

class NMTTargetCharModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder1, decoder2, multigpu=False):
        self.multigpu = multigpu
        super(NMTTargetCharModel, self).__init__()
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.decoder1.bidirectional_encoder = True

    def forward(self, src, tgt, lengths, batch_size, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """

        # Exclude last target (EOS) from input sentences to feed into the word decoder
        tgt1 = torch.stack(tgt.split(tgt.size(1) // batch_size, dim=1), dim=2)
        tgt1 = tgt1[:,:-1,:,:]
        x = (tgt1 != 3).long()*tgt1 + (tgt1 == 3).long()
        tgt1 = x
        tgt1 = torch.cat(tgt1.unbind(2), dim=1)

        # Exclude first target (BOS) from input sentences to feed into the character decoder
        tgt2 = torch.stack(tgt.split(tgt.size(1) // batch_size, dim=1), dim=2)
        tgt2 = tgt2[:,1::,:,:]
        tgt2 = torch.cat(tgt2.unbind(2), dim=1)

        enc_final, memory_bank = self.encoder(src, lengths)
        
        enc_state = \
            self.decoder1.init_decoder_state(src, memory_bank, enc_final)

        attns, ctxs, dec_state, logloss = \
            self.decoder1(tgt1, memory_bank, batch_size,
                         enc_state if dec_state is None
                         else dec_state, 
                         memory_lengths=lengths)

        # Reorder tgt as word_1- batch_1, word1-batch_2 ... word_n-1-batch_n, word_n-batch,n
        newbatch = []
        seqlen = tgt2.size(1) // batch_size
        out = tgt2.split(seqlen, dim=1)
        for k in range(seqlen):
            for l in range(batch_size):
                newbatch.append(out[l][:,k,:])
        tgt2_new = torch.stack(newbatch, dim=1)
        tgt2_new = tgt2_new[:-1,:,:] #exclude the last char from the input
        tgt2_neww = (tgt2_new != 4).long()*tgt2_new + (tgt2_new == 4).long()

        char_state = \
            self.decoder2.init_decoder_state((dec_state.input_feed,), self.decoder1.embeddings.embedding_size)

        
        decoder_outputs, char_state = \
            self.decoder2(tgt2_neww, batch_size,
                         ctxs, char_state)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return decoder_outputs, attns, dec_state, logloss

class CharNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder1, decoder2, multigpu=False):
        self.multigpu = multigpu
        super(CharNMTModel, self).__init__()
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.decoder1.bidirectional_encoder = True

    def forward(self, src, tgt, lengths, batch_size, dec_state=None):

        enc_final, memory_bank = self.encoder(src, batch_size, lengths)
        #enc_state = Variable(enc_final[0].data.new(enc_final[0].size()).zero_().unsqueeze(0).repeat(self.decoder1.num_layers, 1, 2))
        #enc_state = RNNDecoderState(self.decoder1.hidden_size, enc_state)
            #self.decoder.init_decoder_state(src, memory_bank, enc_final)
        enc_state = \
            self.decoder1.init_decoder_state(src, memory_bank, enc_final)

        # Exclude last target (EOS) from input sentences to feed into the word decoder
        tgt1 = torch.stack(tgt.split(tgt.size(1) // batch_size, dim=1), dim=2)
        tgt1 = tgt1[:,:-1,:,:]
        x = (tgt1 != 3).long()*tgt1 + (tgt1 == 3).long()
        tgt1 = x
        tgt1 = torch.cat(tgt1.unbind(2), dim=1)

        # Exclude first target (BOS) from input sentences to feed into the character decoder
        tgt2 = torch.stack(tgt.split(tgt.size(1) // batch_size, dim=1), dim=2)
        tgt2 = tgt2[:,1::,:,:]
        tgt2 = torch.cat(tgt2.unbind(2), dim=1)

        attns, ctxs, dec_state, logloss = \
            self.decoder1(tgt1, memory_bank, batch_size,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths[1])

        # Reorder tgt as word_1- batch_1, word1-batch_2 ... word_n-1-batch_n, word_n-batch,n
        newbatch = []
        seqlen = tgt2.size(1) // batch_size
        out = tgt2.split(seqlen, dim=1)
        for k in range(seqlen):
            for l in range(batch_size):
                newbatch.append(out[l][:,k,:])
        tgt2_new = torch.stack(newbatch, dim=1)
        tgt2_new = tgt2_new[:-1,:,:] #exclude the last char from the input
        tgt2_neww = (tgt2_new != 4).long()*tgt2_new + (tgt2_new == 4).long()

        char_state = \
            self.decoder2.init_decoder_state((dec_state.input_feed,), self.decoder1.embeddings.embedding_size)

        decoder_outputs, char_state = \
            self.decoder2(tgt2_neww, batch_size,
                         ctxs, char_state)

        return decoder_outputs, attns, dec_state, logloss

