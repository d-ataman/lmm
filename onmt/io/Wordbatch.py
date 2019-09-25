import torchtext
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import six

def segment_into_trigrams(inputbatch):
    newbatch = []
    lengths = []
    for line in inputbatch:
        sen = []
        for w in line:
            trigrams = []
            l = int(w.__len__())
            for i in range(0, l):
                if l == 1:
                    trigram = ''.join(('^', w[0], '$'))
                    trigrams.append(trigram)
                else:
                    if i == 0:
                        trigram = ''.join(('^', w[0], w[1]))
                        trigrams.append(trigram)
                    else:
                        if i == l - 1:
                            trigram = ''.join((w[l - 2], w[l - 1], '$'))
                            trigrams.append(trigram)
                        else:
                            trigram = ''.join((w[i - 1], w[i], w[i + 1]))
                            trigrams.append(trigram)
            trigrams.append('$$$')
            sen.append(trigrams)
            lengths += [len(trigrams)]
        newbatch.append(sen)
 
    return tuple(newbatch), lengths


def segment_into_chars(inputbatch):
    newbatch = []
    lengths = []
    for line in inputbatch:
        sen = []
        for w in line:
            chars = ['^^']
            l = int(w.__len__())
            for i in range(0, l):
                char = w[i]
                chars.append(char)
            char = '$$'
            chars.append(char)
            sen.append(chars)
            lengths += [len(chars)]
        newbatch.append(sen)

    return tuple(newbatch), lengths


def split_bpe(inputbatch):
    newbatch = []
    lengths = []
    for line in inputbatch:
        sen = []
        for w in line:
            chars = ['^^']
            char = []
            l = int(w.__len__())
            i = 0
            while i < l:
                if w[i] != '@':
                    char.append(w[i])
                    i += 1
                    if i == l:
                        chars.append(''.join(char))
                else:
                    chars.append(''.join(char))
                    char = []
                    i += 2
            chars.append('$$')
            sen.append(chars)
            lengths += [len(chars)]
        newbatch.append(sen)

    return tuple(newbatch), lengths



class SourceTrigramBatch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None, train=True):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            for (name, field) in dataset.fields.items():
                if field is not None and name == 'src':
                    batch = [x.__dict__[name] for x in data]
                    if name == 'src':
                        newbatch, _ = segment_into_trigrams(batch)
                        idx_batch = [field.process(x, device=device, train=train) for x in newbatch]
                        if isinstance(idx_batch[0], tuple):
                            lengths = [x[1] for x in idx_batch]
                            idx_batch = [x[0] for x in idx_batch]
                            max_char_len = max([x.size(0) for x in idx_batch])
                            seq_lens = [x.size(1) for x in idx_batch]
                            max_seq_len = max(seq_lens)
                            idx_batch = [F.pad(x, (0, max_seq_len - x.size(1), 0, max_char_len-x.size(0)), 'constant', 1) for x in idx_batch]
                            lengths = [F.pad(x, (0,  max_seq_len-x.size(0)), 'constant', max_char_len) for x in lengths]
                        idx_batch = torch.cat(idx_batch, dim=-1)
                        setattr(self, name, (idx_batch, (torch.cat(lengths), torch.cuda.LongTensor(seq_lens))))
                elif field is not None:
                    batch = [x.__dict__[name] for x in data]
                    setattr(self, name, field.process(batch, device=device, train=train))


    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch


class TargetCharBatch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None, train=True):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [x.__dict__[name] for x in data]
                    if name == 'tgt':
                        #newbatch, _ = segment_into_chars(batch)
                        #newbatch, _ = segment_into_trigrams(batch)
                        newbatch, _ = split_bpe(batch)
                        idx_batch = [field.process(x, device=device, train=train) for x in newbatch]

                        # Pad the sentences and add BOS and EOS tokens
                        max_char_len = max([x.size(0) for x in idx_batch])
                        seq_lens = [x.size(1) for x in idx_batch]
                        max_seq_len = max(seq_lens)
                        idx_batch = [torch.cat([torch.tensor([[2,2]+[1]*(x.size(0)-2)]).transpose(0,1).cuda(), x], dim=1) for x in idx_batch]
                        idx_batch = [torch.cat([x, torch.tensor([[5,5,3,4]+[1]*(x.size(0)-4)]).transpose(0,1).cuda()], dim=1) for x in idx_batch]  
                        idx_batch = [F.pad(x[1::], (0, max_seq_len+2-x.size(1), 0, max_char_len-x.size(0)), 'constant', 1) for x in idx_batch]
                        for x in idx_batch:
                            for i in range(x.size(1)):
                                for j in range(x.size(0)):
                                    if x[j,i] == 3 and j != 1:
                                        x[j,i] = 1
                        idx_batch = torch.cat(idx_batch, dim=-1)
                        setattr(self, name, idx_batch)
                    else:
                        setattr(self, name, field.process(batch, device=device, train=train))
                        


    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch

# Doesn't work yet!!
class BothTrigramBatch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None, train=True):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            for (name, field) in dataset.fields.items():
                if field is not None and name in ['src', 'tgt']:
                    batch = [x.__dict__[name] for x in data]

                    if name == 'src':
                        #newbatch, _ = segment_into_trigrams(batch)
                        newbatch, _ = segment_into_chars(batch)
                        idx_batch = [field.process(x, device=device, train=train) for x in newbatch]
                        if isinstance(idx_batch[0], tuple):
                            lengths = [x[1] for x in idx_batch]
                            idx_batch = [x[0] for x in idx_batch]
                            max_char_len = max([x.size(0) for x in idx_batch])
                            seq_lens = [x.size(1) for x in idx_batch]
                            max_seq_len = max(seq_lens)
                            idx_batch = [F.pad(x, (0, max_seq_len - x.size(1), 0, max_char_len-x.size(0)), 'constant', 1) for x in idx_batch]
                            lengths = [F.pad(x, (0,  max_seq_len-x.size(0)), 'constant', max_char_len) for x in lengths]
                        idx_batch = torch.cat(idx_batch, dim=-1)
                        setattr(self, name, (idx_batch, (torch.cat(lengths), torch.cuda.LongTensor(seq_lens))))

                    elif name == 'tgt':
                        newbatch, _ = split_bpe(batch)
                        idx_batch = [field.process(x, device=device, train=train) for x in newbatch]

                        # Pad the sentences and add BOS and EOS tokens
                        max_char_len = max([x.size(0) for x in idx_batch])
                        seq_lens = [x.size(1) for x in idx_batch]
                        max_seq_len = max(seq_lens)
                        idx_batch = [torch.cat([torch.tensor([[2,2]+[1]*(x.size(0)-2)]).transpose(0,1).cuda(), x], dim=1) for x in idx_batch]
                        idx_batch = [torch.cat([x, torch.tensor([[5,5,3,4]+[1]*(x.size(0)-4)]).transpose(0,1).cuda()], dim=1) for x in idx_batch]
                        idx_batch = [F.pad(x[1::], (0, max_seq_len+2-x.size(1), 0, max_char_len-x.size(0)), 'constant', 1) for x in idx_batch]
                        for x in idx_batch:
                            for i in range(x.size(1)):
                                for j in range(x.size(0)):
                                    if x[j,i] == 3 and j != 1:
                                        x[j,i] = 1
                        idx_batch = torch.cat(idx_batch, dim=-1)
                        setattr(self, name, idx_batch)

                elif field is not None:
                    batch = [x.__dict__[name] for x in data]
                    setattr(self, name, field.process(batch, device=device, train=train))

    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch



class SourceWordIterator(torchtext.data.Iterator):

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=None, shuffle=None, sort=None,
                 sort_within_batch=None):
        super(SourceWordIterator, self).__init__(dataset, batch_size, sort_key, device,
                                           batch_size_fn, train, repeat, shuffle,
                                           sort, sort_within_batch)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield SourceTrigramBatch(minibatch, self.dataset, self.device,
                            self.train)
            if not self.repeat:
                raise StopIteration

class TargetWordIterator(torchtext.data.Iterator):

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=None, shuffle=None, sort=None,
                 sort_within_batch=None):
        super(TargetWordIterator, self).__init__(dataset, batch_size, sort_key, device,
                                           batch_size_fn, train, repeat, shuffle,
                                           sort, sort_within_batch)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield TargetCharBatch(minibatch, self.dataset, self.device,
                            self.train)
            if not self.repeat:
                raise StopIteration

class BothWordIterator(torchtext.data.Iterator):

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=None, shuffle=None, sort=None,
                 sort_within_batch=None):
        super(BothWordIterator, self).__init__(dataset, batch_size, sort_key, device,
                                           batch_size_fn, train, repeat, shuffle,
                                           sort, sort_within_batch)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield BothTrigramBatch(minibatch, self.dataset, self.device,
                            self.train)
            if not self.repeat:
                raise StopIteration

