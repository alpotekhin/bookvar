"""Execute every published block; check the numerical representation contracts."""
from pathlib import Path
import math
import re
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/"00 Учебник/02 Представление текста и токенизация/01 Представление текста числами.md"
text=CHAPTER.read_text()
assert not re.search(r'^\$$',text,re.M)
assert chr(0xA4) not in text
blocks=re.findall(r'```python\n(.*?)```',text,re.S)
assert len(blocks)==5
scopes=[]
for i,code in enumerate(blocks,1):
    scope={}
    exec(compile(code,f"{CHAPTER}:block{i}","exec"),scope)
    scopes.append(scope)
    print("PASS published block",i)
unicode,vocabulary,lookup,batch,tfidf=scopes
assert len("кот спит")==8 and len("кот спит".encode("utf8"))==15
assert "к".encode("utf8")==bytes.fromhex("d0ba")
assert [ord(c) for c in "👩‍💻"]==[0x1F469,0x200D,0x1F4BB]
assert [len(s.encode("utf8")) for s in ("é","e\u0301")]==[2,3]
counts=vocabulary["Counter"](w for doc in vocabulary["corpus"] for w in doc.split())
assert sum(counts.values())==8 and dict(counts)=={"кот":3,"спит":3,"пёс":1,"бежит":1}
assert vocabulary["encode"]("кот  спит")==vocabulary["encode"]("кот спит")
assert vocabulary["decode"](vocabulary["encode"]("лиса тигр"))==["<unk>","<unk>"]
assert vocabulary["vocab"]["<pad>"]==0 and vocabulary["vocab"]["<unk>"]==1
# Reserved spellings do not produce duplicate vocabulary entries.
extra=vocabulary["build_vocab"](["<pad> кот <unk>"])[1]
assert extra==["<pad>","<unk>","кот"]

dtype=torch.float64
weights=lookup["weights"]
onehot=F.one_hot(torch.arange(6),num_classes=6).to(dtype)
torch.testing.assert_close(onehot@onehot.T,torch.eye(6,dtype=dtype))
torch.testing.assert_close((onehot[2]-onehot[4]).square().sum(),torch.tensor(2.,dtype=dtype))
assert F.cosine_similarity(onehot[2],onehot[4],dim=0).item()==0

# Vocabulary permutation preserves lookup and the output distribution under relabeling.
ids=torch.tensor([2,3,2])
new_to_old=torch.tensor([0,1,4,3,2,5])
old_to_new=torch.argsort(new_to_old)
new_weights=weights[new_to_old]
hidden=F.embedding(ids,weights)
torch.testing.assert_close(F.embedding(old_to_new[ids],new_weights),hidden)
old_probs=(hidden@weights.T).softmax(-1)
new_probs=(hidden@new_weights.T).softmax(-1)
torch.testing.assert_close(new_probs,old_probs[:,new_to_old])
assert 4*1024*50000*4/2**20==781.25
assert 4*1024*8/2**10==32
assert 4*1024*768*4/2**20==12
np.testing.assert_allclose(50000*768*4/2**20,146.484375)

# Repeated IDs sum gradients; frequency scaling divides by their minibatch frequency.
incoming=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=dtype)
assert torch.autograd.gradcheck(lambda w:F.embedding(ids,w),(weights.clone().requires_grad_(),))
frequency=torch.nn.Embedding.from_pretrained(weights.clone(),freeze=False,scale_grad_by_freq=True)
(frequency(ids)*incoming).sum().backward()
torch.testing.assert_close(frequency.weight.grad[2],torch.tensor([3.,4.],dtype=dtype))
torch.testing.assert_close(frequency.weight.grad[3],torch.tensor([3.,4.],dtype=dtype))
sparse=torch.nn.Embedding.from_pretrained(weights.clone(),freeze=False,sparse=True)
(sparse(ids)*incoming).sum().backward()
assert not sparse.weight.is_sparse and sparse.weight.grad.is_sparse
assert sparse.weight.numel()==12 and sparse.weight.grad.coalesce()._nnz()==2
expected_grad=torch.zeros_like(weights)
expected_grad[2]=torch.tensor([6.,8.]);expected_grad[3]=torch.tensor([3.,4.])
torch.testing.assert_close(sparse.weight.grad.to_dense(),expected_grad)
decay=torch.nn.Embedding.from_pretrained(weights.clone(),freeze=False)
(decay(ids)*incoming).sum().backward()
torch.optim.SGD(decay.parameters(),lr=.1,weight_decay=.1).step()
torch.testing.assert_close(decay.weight[4],weights[4]*.99)

# Padding freezes its lookup gradient, not its loaded numerical value.
loaded=weights.clone();loaded[0]=torch.tensor([5.,6.]);loaded.requires_grad_()
padded_ids=batch["ids"]
result=F.embedding(padded_ids,loaded,padding_idx=0)
torch.testing.assert_close(result[1,2],torch.tensor([5.,6.],dtype=dtype))
result.sum().backward()
torch.testing.assert_close(loaded.grad[0],torch.zeros(2,dtype=dtype))
for width in (3,5,8):
    extended=F.pad(padded_ids,(0,width-3),value=0)
    valid=extended!=0;x=F.embedding(extended,weights)
    pooled=(x*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True)
    torch.testing.assert_close(pooled,batch["expected"])
assert torch.equal(batch["valid"],torch.tensor([[True,True,True],[True,True,False]]))

# BOW, bigrams and their limits, including the single-character default-token trap.
assert CountVectorizer().build_analyzer()("Я люблю NLP")==["люблю","nlp"]
def vectorizer(**kw):
    return CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,**kw)
names=["John loves Mary","Mary loves John"]
bow=vectorizer().fit_transform(names).toarray()
np.testing.assert_array_equal(bow[0],bow[1])
pairs=vectorizer(ngram_range=(1,2)).fit_transform(names).toarray()
assert not np.array_equal(pairs[0],pairs[1])
counterexample=vectorizer(ngram_range=(1,2)).fit_transform(["a b a c a","a c a b a"]).toarray()
np.testing.assert_array_equal(counterexample[0],counterexample[1])

C=tfidf["counts"];dense=C.toarray()
np.testing.assert_array_equal((dense>0).sum(0),[2,3,1,1])
assert C.nnz==7 and (C.data.nbytes+C.indices.nbytes+C.indptr.nbytes)>dense.nbytes
manual_idf=np.log(3/np.array([2,3,1,1]))
np.testing.assert_allclose(dense[0]*manual_idf,[2*np.log(3/2),0,0,0])
np.testing.assert_allclose(2*np.log(3/2),.8109302162163288)
default=tfidf["transformer"]
np.testing.assert_allclose(default.idf_,[1.2876820724517808,1,1.6931471805599454,1.6931471805599454])
raw=dense[0]*default.idf_
np.testing.assert_allclose(np.linalg.norm(raw),2.7627,atol=5e-5)
np.testing.assert_allclose(tfidf["weighted"].toarray()[0],[.93219169,.361965,0,0],atol=1e-8)
np.testing.assert_allclose(np.linalg.norm(tfidf["weighted"].toarray(),axis=1),[1,1,1])
unsmoothed=TfidfTransformer(smooth_idf=False,norm=None).fit(C)
np.testing.assert_allclose(unsmoothed.idf_,1+manual_idf)
sublinear=TfidfTransformer(sublinear_tf=True,norm=None).fit_transform(C).toarray()
np.testing.assert_allclose(sublinear[0,0],(1+np.log(2))*default.idf_[0])
before=default.idf_.copy();unknown=tfidf["vectorizer"].transform(["тигр"])
assert unknown.nnz==0 and default.transform(unknown).nnz==0
np.testing.assert_array_equal(default.idf_,before)
v=torch.tensor([2.,1.,0.,0.],dtype=dtype)
torch.testing.assert_close(F.cosine_similarity(v,2*v,dim=0),torch.tensor(1.,dtype=dtype))
assert np.array_equal(dense[0]>0,np.array([True,True,False,False]))
print("PASS Unicode/UTF8/NFC; vocabulary/OOV/roundtrip/permutation; one-hot/memory; "
      "lookup/VJP/frequency/sparse/decay; padding/masked pooling; BOW/bigrams/CSR; "
      "manual/sklearn TF-IDF, smoothing/sublinear/zero-vector and cosine. torch",torch.__version__)
