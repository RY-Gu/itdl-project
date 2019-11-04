import torch
from collections import Counter
from torchtext import data
import pargs as arg
from copy import copy


class dataset:
    def __init__(self, args):
        args.path = args.datadir + args.data
        print("Loading Data from ", args.path)
        self.args = args
        self.mkVocabs(args)
        print("\nVocab Sizes:")
        for x in self.fields:
            try:
                print(x[0], len(x[1].vocab))
            except:
                try:
                    print(x[0], len(x[1].itos))
                except:
                    pass

    def build_ent_vocab(self, path, unkat=0):
        ents = ""
        with open(path, encoding='utf-8') as f:
            for l in f:
                ents += " " + l.split("\t")[1]
        itos = sorted(list(set(ents.split(" "))))
        itos[0] == "<unk>";
        itos[1] == "<pad>"
        stoi = {x: i for i, x in enumerate(itos)}
        return itos, stoi

    def vec_ents(self, ex, field):
        # returns tensor and lens
        ex = [[field.stoi[x] if x in field.stoi else 0 for x in y.strip().split(" ")] for y in ex.split(";")]
        return self.pad_list(ex, 1)

    def mkGraphs(self, r, ent):
        '''
        :param r: relation string of a dataset row
        :param ent: number of entities in a dataset row
        :return: adj, rel matrices
        '''

        # convert triples to entlist with adj and rel matrices
        triples = r.strip().split(';')
        x = [[int(y) for y in triple.strip().split()] for triple in triples]
        rel = [2]  # global root node (i.e. 'ROOT')
        adjsize = ent + 1 + (2 * len(x))
        adj = torch.zeros(adjsize, adjsize)  # adjacency matrix (# Entity + Global root + # relations) ^2
        for i in range(ent):
            adj[i, ent] = 1
            adj[ent, i] = 1
        for i in range(adjsize):
            adj[i, i] = 1
        for y in x:
            rel.extend([y[1] + 3, y[1] + 3 + self.REL.size])
            a = y[0]
            b = y[2]
            c = ent + len(rel) - 2
            d = ent + len(rel) - 1
            adj[a, c] = 1
            adj[c, b] = 1
            adj[b, d] = 1
            adj[d, a] = 1
        rel = torch.LongTensor(rel)  # rel: ['ROOT', 'REL 1', 'REL 1_inv', 'REL 2', 'REL 2_inv', ...]
        return (adj, rel)

    def adjToSparse(self, adj):
        sp = []
        for row in adj:
            sp.append(row.nonzero().squeeze(1))
        return sp

    def mkVocabs(self, args):
        args.path = args.datadir + args.data
        self.INP = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>",
                              include_lengths=True)  # Title
        self.OUTP = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>",
                               include_lengths=True)  # Gold Abstract, preprocessed
        self.TGT = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>")
        self.NERD = data.Field(sequential=True, batch_first=True, eos_token="<eos>")  # Entity Type
        self.ENT = data.RawField()  # Entity
        self.REL = data.RawField()  # Relation between entities
        self.SORDER = data.RawField()
        self.SORDER.is_target = False
        self.REL.is_target = False
        self.ENT.is_target = False
        self.fields = [("src", self.INP), ("ent", self.ENT), ("nerd", self.NERD), ("rel", self.REL), ("out", self.OUTP),
                       ("sorder", self.SORDER)]
        train = data.TabularDataset(path=args.path, format='tsv', fields=self.fields)

        print('Building Vocab... ', end='')

        # Output Vocab
        # generics => indices are at the last of the vocab
        # also includes indexed generics, (e.g. <method_0>)
        self.OUTP.build_vocab(train, min_freq=args.outunk)
        generics = ['<method>', '<material>', '<otherscientificterm>', '<metric>', '<task>']  # Entity Types
        self.OUTP.vocab.itos.extend(generics)
        for x in generics:
            self.OUTP.vocab.stoi[x] = self.OUTP.vocab.itos.index(x)

        # Target Vocab
        # Same as Output Vocab, except for the indexed generics' indices
        # len(vocab) = 11738 / <method_0>, <material_0> ... : 11738, <method_1>, ... : 11739 and so on.
        self.TGT.vocab = copy(self.OUTP.vocab)
        specials = "method material otherscientificterm metric task".split(" ")
        for x in specials:
            for y in range(40):
                s = "<" + x + "_" + str(y) + ">"
                self.TGT.vocab.stoi[s] = len(self.TGT.vocab.itos) + y

        # Entity Type Vocab
        # Indices for not-indexed generics are same with those of output vocab
        self.NERD.build_vocab(train, min_freq=0)
        for x in generics:
            self.NERD.vocab.stoi[x] = self.OUTP.vocab.stoi[x]

        # Title Vocab
        self.INP.build_vocab(train, min_freq=args.entunk)

        # Relation Vocab
        # Adds relations.vocab + inverse of relations.vocab
        self.REL.special = ['<pad>', '<unk>', 'ROOT']
        with open(args.datadir + "/" + args.relvocab) as f:
            rvocab = [x.strip() for x in f.readlines()]
            self.REL.size = len(rvocab)
            rvocab += [x + "_inv" for x in rvocab]
            relvocab = self.REL.special + rvocab
        self.REL.itos = relvocab

        self.ENT.itos, self.ENT.stoi = self.build_ent_vocab(args.path)

        print('Done')
        if not self.args.eval:
            self.mkiters(train)

    def listTo(self, l):
        return [x.to(self.args.device) for x in l]

    def fixBatch(self, b):

        ent, phlens = zip(*b.ent)  # b.ent: (batch_size,) / list of x.ent
        # ent: (batch_size,) / tuple of (num of entity, longest entity len) per each dataset row
        # phlens: (batch_size,) / tuple of (num of entity) per each dataset row i.e. 각 row의 각 entity의 단어 개수
        ent, elens = self.adjToBatch(ent)
        # ent: (sum of num of entities in each row, longest entity len)
        # elens: (batch_size,) / tensor of number of entities in each row
        ent = ent.to(self.args.device)

        adj, rel = zip(*b.rel)  # b.rel: (batch_size,) / list of x.rel
        # adj: (batch_size,) / tuple of adjacency matrices per each dataset row
        # rel: (batch_size,) / tuple of list of relations per each dataset row
        if self.args.sparse:
            b.rel = [adj, self.listTo(rel)]
        else:
            b.rel = [self.listTo(adj), self.listTo(rel)]  # [[adj1, adj2, ...], [rel1, rel2, ...]]
        if self.args.plan:
            b.sordertgt = self.listTo(self.pad_list(b.sordertgt))
        phlens = torch.cat(phlens, 0).to(self.args.device)
        # phlens: (sum of num of entities in each row, ) / tensor of 각 entity의 단어 개수

        elens = elens.to(self.args.device)
        b.ent = (ent, phlens, elens)
        return b

    def adjToBatch(self, adj):
        lens = [x.size(0) for x in adj]  # lens: list of # of entities in each dataset row
        m = max([x.size(1) for x in adj])  # m: longest entity len in the batch
        data = [self.pad(x.transpose(0, 1), m).transpose(0, 1) for x in adj]
        data = torch.cat(data, 0)
        return data, torch.LongTensor(lens)  # data: pads each x in adj to (,m), and concat them

    def bszFn(self, e, l, c):
        return c + len(e.out)

    def mkiters(self, train):
        args = self.args
        c = Counter([len(x.out) for x in train])
        t1, t2, t3 = [], [], []
        print("Sorting training data by len")
        for x in train:
            l = len(x.out)
            if l < 100:
                t1.append(x)
            elif l > 100 and l < 220:
                t2.append(x)
            else:
                t3.append(x)
        t1d = data.Dataset(t1, self.fields)
        t2d = data.Dataset(t2, self.fields)
        t3d = data.Dataset(t3, self.fields)
        valid = data.TabularDataset(path=args.path.replace("train", "val"), format='tsv', fields=self.fields)
        print("Dataset Sizes (t1, t2, t3, valid):", end=' ')
        for ds in [t1d, t2d, t3d, valid]:
            print(len(ds.examples), end=' ')
            for x in ds:
                x.rawent = x.ent.split(" ; ")
                x.ent = self.vec_ents(x.ent, self.ENT)
                # x.ent: (num of entity, longest entity len), (num of entity)

                x.rel = self.mkGraphs(x.rel, len(x.ent[1]))
                if args.sparse:
                    x.rel = (self.adjToSparse(x.rel[0]), x.rel[1])

                x.tgt = x.out
                x.out = [y.split("_")[0] + ">" if "_" in y else y for y in x.out]
                # x.out: removes tag indices for out (e.g. <method_0> => <method>

                x.sordertgt = torch.LongTensor([int(y) + 3 for y in x.sorder.split(" ")])
                x.sorder = [[int(z) for z in y.strip().split(" ")] for y in x.sorder.split("-1")[:-1]]
            ds.fields["tgt"] = self.TGT
            ds.fields["rawent"] = data.RawField()
            ds.fields["sordertgt"] = data.RawField()
            ds.fields["rawent"].is_target = False
            ds.fields["sordertgt"].is_target = False

        self.t1_iter = data.Iterator(t1d, args.t1size, device=args.device, sort_key=lambda x: len(x.out), repeat=False,
                                     train=True)
        self.t2_iter = data.Iterator(t2d, args.t2size, device=args.device, sort_key=lambda x: len(x.out), repeat=False,
                                     train=True)
        self.t3_iter = data.Iterator(t3d, args.t3size, device=args.device, sort_key=lambda x: len(x.out), repeat=False,
                                     train=True)
        self.val_iter = data.Iterator(valid, args.t3size, device=args.device, sort_key=lambda x: len(x.out), sort=False,
                                      repeat=False, train=False)

    def mktestset(self, args):
        path = args.path.replace("train", 'test')
        fields = self.fields
        ds = data.TabularDataset(path=path, format='tsv', fields=fields)
        ds.fields["rawent"] = data.RawField()
        ds.fields["rawent"].is_target = False
        for x in ds:
            x.rawent = x.ent.split(" ; ")
            x.ent = self.vec_ents(x.ent, self.ENT)  # tuple of ((# of entities in x, max entity len), (# of entities))
            x.rel = self.mkGraphs(x.rel, len(x.ent[1]))  # tuple of (adj, rel)
            if args.sparse:
                x.rel = (self.adjToSparse(x.rel[0]), x.rel[1])
            x.tgt = x.out
            x.out = [y.split("_")[0] + ">" if "_" in y else y for y in x.out]
            x.sordertgt = torch.LongTensor([int(y) + 3 for y in x.sorder.split(" ")])
            x.sorder = [[int(z) for z in y.strip().split(" ")] for y in x.sorder.split("-1")[:-1]]
        ds.fields["tgt"] = self.TGT
        ds.fields["rawent"] = data.RawField()
        ds.fields["sordertgt"] = data.RawField()
        ds.fields["rawent"].is_target = False
        ds.fields["sordertgt"].is_target = False
        dat_iter = data.Iterator(ds, 1, device=args.device, sort_key=lambda x: len(x.src), train=False, sort=False)
        return dat_iter

    def reverse(self, x, ents):
        ents = ents[0]
        vocab = self.TGT.vocab
        s = ' '.join(
            [vocab.itos[y] if y < len(vocab.itos) else ents[y - len(vocab.itos)].upper() for j, y in enumerate(x)])

        if "<eos>" in s: s = s.split("<eos>")[0]
        return s

    def pad_list(self, l, ent=1):
        lens = [len(x) for x in l]
        m = max(lens)
        return torch.stack([self.pad(torch.tensor(x), m, ent) for x in l], 0), torch.LongTensor(lens)

    def pad(self, tensor, length, ent=1):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(ent)])

    # def rev_ents(self, batch):
    #     vocab = self.NERD.vocab
    #     es = []
    #     for e in batch:
    #         s = [vocab.itos[y].split(">")[0] + "_" + str(i) + ">" for i, y in enumerate(e) if
    #              vocab.itos[y] not in ['<pad>', '<eos>']]
    #         es.append(s)
    #     return es

    # def relfix(self, relstrs):
    #     mat = []
    #     for x in relstrs:
    #         pieces = x.strip().split(';')
    #         x = [[int(y) + len(self.REL.special) for y in z.strip().split()] for z in pieces]
    #         mat.append(torch.LongTensor(x).cuda())
    #     lens = [x.size(0) for x in mat]
    #     m = max(lens)
    #     mat = [self.pad(x, m) for x in mat]
    #     mat = torch.stack(mat, 0)
    #     lens = torch.LongTensor(lens).cuda()
    #     return mat, lens
    #
    # def getEnts(self, entseq):
    #     newents = []
    #     lens = []
    #     for i, l in enumerate(entseq):
    #         l = l.tolist()
    #         if self.enteos in l:
    #             l = l[:l.index(self.enteos)]
    #         tmp = []
    #         while self.entspl in l:
    #             tmp.append(l[:l.index(self.entspl)])
    #             l = l[l.index(self.entspl) + 1:]
    #         if l:
    #             tmp.append(l)
    #         lens.append(len(tmp))
    #         tmplen = [len(x) for x in tmp]
    #         m = max(tmplen)
    #         tmp = [x + ([1] * (m - len(x))) for x in tmp]
    #         newents.append((torch.LongTensor(tmp).cuda(), torch.LongTensor(tmplen).cuda()))
    #     return newents, torch.LongTensor(lens).cuda()
    #
    # def listToBatch(self, inp):
    #     data, lens = zip(*inp)
    #     print(lens);
    #     exit()
    #     lens = torch.tensor(lens)
    #     m = torch.max(lens).item()
    #     data = [self.pad(x.transpose(0, 1), m).transpose(0, 1) for x in data]
    #     data = torch.cat(data, 0)
    #     return data, lens
    #
    # def rev_rel(self, ebatch, rbatch):
    #     vocab = self.ENT.vocab
    #     for i, ents in enumerate(ebatch):
    #         es = []
    #         for e in ents:
    #             s = ' '.join([vocab.itos[y] for y in e])
    #             es.append(s)
    #         rels = rbatch[i]
    #         for a, r, b in rels:
    #             print(es[a], self.REL.itos[r], es[b])
    #         print()

    # def seqentmat(self, entseq):
    #     newents = []
    #     lens = []
    #     sms = []
    #     for l in entseq:
    #         l = l.tolist()
    #         if self.enteos in l:
    #             l = l[:l.index(self.enteos)]
    #         tmp = []
    #         while self.entspl in l:
    #             tmp.append(l[:l.index(self.entspl)])
    #             l = l[l.index(self.entspl) + 1:]
    #         if l:
    #             tmp.append(l)
    #         lens.append(len(tmp))
    #         m = max([len(x) for x in tmp])
    #         sms.append(m)
    #         tmp = [x + ([0] * (m - len(x))) for x in tmp]
    #         newents.append(tmp)
    #     sm = max(lens)
    #     pm = max(sms)
    #     for i in range(len(newents)):
    #         tmp = torch.LongTensor(newents[i]).transpose(0, 1)
    #         tmp = self.pad(tmp, pm, ent=0)
    #         tmp = tmp.transpose(0, 1)
    #         tmp = self.pad(tmp, sm, ent=0)
    #         newents[i] = tmp
    #     newents = torch.stack(newents, 0).cuda()
    #     lens = torch.LongTensor(lens).cuda()
    #     return newents, lens


if __name__ == "__main__":
    args = arg.pargs()
    ds = dataset(args)
    ds.getBatch()