"""Check the chapter's mechanisms, three code blocks and preprocessing isolation."""
from pathlib import Path
from itertools import product, permutations
import hashlib
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score, mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    silhouette_score, silhouette_samples, accuracy_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/06 Кластеризация и её ограничения.md"
blocks = re.findall(r"```python\n(.*?)```", CHAPTER.read_text(), re.S)
assert len(blocks) == 3
scope = {}
for i,code in enumerate(blocks[:2]):
    exec(compile(code,str(CHAPTER)+f":block{i+1}","exec"),scope)
np.testing.assert_allclose(scope["history"],[[150,39.25],[13.9375,4]])
np.testing.assert_allclose(scope["centers"],[1,9])

# Both assignment and mean updates decrease exactly the displayed objective.
x = np.array([0,1,2,8,9,10.])
old_centers = np.array([0,2.])
labels = ((x[:,None]-old_centers)**2).argmin(axis=1)
np.testing.assert_array_equal(labels,[0,0,1,1,1,1])
new_centers = np.array([x[labels==j].mean() for j in range(2)])
np.testing.assert_allclose(new_centers,[.5,7.25])
assert np.mean(new_centers)==3.875
for j in range(2):
    cluster=x[labels==j]; mean=cluster.mean()
    for proposed in [-10,-.3,2,15]:
        np.testing.assert_allclose(np.sum((cluster-proposed)**2),
          np.sum((cluster-mean)**2)+len(cluster)*(proposed-mean)**2)

# A genuine local fixed point, and exhaustive global optimum over all assignments.
local_x=np.array([0,1,4,5,9.])[:,None]
local=KMeans(2,init=np.array([[0.],[9.]]),n_init=1,algorithm="lloyd").fit(local_x)
better=KMeans(2,init=np.array([[0.],[4.]]),n_init=1,algorithm="lloyd").fit(local_x)
np.testing.assert_allclose(local.cluster_centers_.ravel(),[5/3,7])
np.testing.assert_allclose(local.inertia_,50/3)
np.testing.assert_allclose(better.cluster_centers_.ravel(),[.5,6])
np.testing.assert_allclose(better.inertia_,14.5)
objectives=[]
for assignment in product([0,1],repeat=5):
    z=np.array(assignment)
    if len(set(z))<2: continue
    objectives.append(sum(np.sum((local_x[z==j]-local_x[z==j].mean())**2) for j in range(2)))
np.testing.assert_allclose(min(objectives),14.5)
np.testing.assert_allclose(x*x,[0,1,4,64,81,100])
assert np.sum(x*x)==250 and np.sum(x[3:]**2)/250==.98
auto=KMeans(2,n_init="auto",random_state=0).fit(x[:,None])
assert auto._n_init==1
assert KMeans(2,n_init=10,random_state=0).fit(x[:,None])._n_init==10

# Units, block weights, norm identity and the mean's unchanged outlier sensitivity.
A=np.array([1,1000.]); B=np.array([2,1000.]); C=np.array([1,1100.])
np.testing.assert_allclose([np.linalg.norm(A-B),np.linalg.norm(A-C)],[1,100])
scale=np.array([1,1000.])
np.testing.assert_allclose([np.linalg.norm((A-B)/scale),np.linalg.norm((A-C)/scale)],[1,.1])
raw=np.stack([A,B,C])
np.testing.assert_allclose(StandardScaler().fit_transform(raw),StandardScaler().fit_transform(raw/scale))
assert np.sum((np.array([1,0])-np.array([0,1]))**2)==2
assert np.sum((10*np.array([1,0])-10*np.array([0,1]))**2)==200
u=np.array([1.,0]); v=np.array([0.,1])
np.testing.assert_allclose(np.sum((u-v)**2),2*(1-u@v))
assert not np.isclose(np.linalg.norm((u+v)/2),1)
# Unit observations do not force their means to have unit norm. The next
# assignment step can differ after normalizing the centers themselves.
group_a = np.array([[.2, np.sqrt(.96)], [.2, -np.sqrt(.96)]])
group_b = np.array([[.8, .6], [.8, .6]])
np.testing.assert_allclose(np.linalg.norm(np.vstack([group_a, group_b]), axis=1), 1)
means = np.vstack([group_a.mean(axis=0), group_b.mean(axis=0)])
np.testing.assert_allclose(means, [[.2, 0], [.8, .6]])
np.testing.assert_allclose(np.linalg.norm(means, axis=1), [.2, 1])
query_unit = np.array([1., 0])
squared_distances = np.sum((means - query_unit)**2, axis=1)
np.testing.assert_allclose(squared_distances, [.64, .4])
assert squared_distances.argmin() == 1
unit_centers = means / np.linalg.norm(means, axis=1, keepdims=True)
cosines = unit_centers @ query_unit
np.testing.assert_allclose(cosines, [1, .8])
assert cosines.argmax() == 0
np.testing.assert_allclose(squared_distances,
    1 + np.sum(means**2, axis=1) - 2 * (means @ query_unit))
# The normalized sum maximizes total cosine similarity for a fixed cluster.
rng_sphere = np.random.default_rng(20261025)
for dim in [2, 3, 8]:
    for _ in range(80):
        observations = rng_sphere.normal(size=(7, dim))
        observations /= np.linalg.norm(observations, axis=1, keepdims=True)
        total = observations.sum(axis=0)
        optimal = total / np.linalg.norm(total)
        mean_direction = observations.mean(axis=0)
        mean_direction /= np.linalg.norm(mean_direction)
        np.testing.assert_allclose(optimal, mean_direction)
        candidates = rng_sphere.normal(size=(30, dim))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        np.testing.assert_allclose((observations @ optimal).sum(), np.linalg.norm(total))
        assert np.all(candidates @ total <= np.linalg.norm(total) + 1e-12)
opposites = np.array([[1., 0], [-1., 0]])
np.testing.assert_array_equal(opposites.sum(axis=0), [0, 0])
for direction in [[1., 0], [0., 1], [.6, .8]]:
    np.testing.assert_allclose((opposites @ direction).sum(), 0)
print("PASS spherical update: 240 fixed clusters, 7200 candidate directions and zero-sum case")
outliers=np.array([8,9,10,100.])[:,None]
assert outliers.mean()==31.75
scaler=RobustScaler().fit(outliers)
np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(outliers).mean(axis=0,keepdims=True)),[[31.75]])

# Actual neighborhoods, boundary ambiguity, and why border chains do not merge cores.
xx=np.array([0,.4,.8,1.7,4.])[:,None]
counts=(cdist(xx,xx)<=1).sum(axis=1)
np.testing.assert_array_equal(counts,[3,3,4,2,1])
model=DBSCAN(eps=1,min_samples=3).fit(xx)
np.testing.assert_array_equal(model.labels_,[0,0,0,0,-1])
np.testing.assert_array_equal(model.core_sample_indices_,[0,1,2])
assert len(set(model.labels_))==2 and len(set(model.labels_)-{-1})==1
bridge=np.array([-.3,-.2,-.1,0,1,2,2.1,2.2,2.3])[:,None]
forward=DBSCAN(eps=1,min_samples=4).fit(bridge)
backward=DBSCAN(eps=1,min_samples=4).fit(bridge[::-1])
back_labels=backward.labels_[::-1]
assert 4 not in forward.core_sample_indices_
assert forward.labels_[4]==forward.labels_[0] and back_labels[4]==back_labels[-1]
assert forward.labels_[0]!=forward.labels_[-1]
assert adjusted_rand_score(forward.labels_[np.arange(9)!=4],back_labels[np.arange(9)!=4])==1
assert connected_components(cdist(bridge,bridge)<=1,directed=False)[0]==1
assert np.all(DBSCAN(eps=1,min_samples=3).fit_predict([[0],[.9],[5]])==-1)
assert DBSCAN(eps=1,min_samples=3).fit_predict([[0],[0],[.9],[5]])[0]!=-1

# Mutual reachability and equality of all threshold component partitions of MST/full graph.
six=np.array([0,.1,.2,1,1.1,1.2])[:,None]
dist=cdist(six,six)
core=np.sort(dist,axis=1)[:,2]  # Third neighbor INCLUDING self.
np.testing.assert_allclose(core,[.2,.1,.2,.2,.1,.2])
mutual=np.maximum(np.maximum(core[:,None],core[None,:]),dist)
np.testing.assert_allclose([mutual[0,1],mutual[2,3]],[.2,.8])
np.fill_diagonal(mutual,0)
mst=minimum_spanning_tree(mutual).toarray()
np.testing.assert_allclose(np.sort(mst[mst>0]),[.2,.2,.2,.2,.8])
for threshold in [0,.1,.15,.2,.25,.3,.79,.8,.81,1]:
    full=(mutual>0)&(mutual<=threshold)
    tree=(mst>0)&(mst<=threshold)
    full_count,full_labels=connected_components(full,directed=False)
    tree_count,tree_labels=connected_components(tree,directed=False)
    assert full_count==tree_count
    assert adjusted_rand_score(full_labels,tree_labels)==1
assert connected_components((mutual>0)&(mutual<=.3),directed=False)[0]==2
assert connected_components((mutual>0)&(mutual<=1),directed=False)[0]==1
hb=HDBSCAN(min_cluster_size=3,min_samples=3).fit(six)
assert adjusted_rand_score(hb.labels_,[0,0,0,1,1,1])==1
assert adjusted_rand_score(hb.labels_,HDBSCAN(min_cluster_size=3).fit_predict(six))==1
assert HDBSCAN().allow_single_cluster is False
parent=6*(2-1)
children_persistent=[3*(4-2),3*(4-2)]
children_short=[3*(2.2-2),3*(2.2-2)]
np.testing.assert_allclose([parent,sum(children_persistent),sum(children_short)],[6,12,1.2])
assert sum(children_persistent)>parent>sum(children_short)

# Silhouette uses other objects, not the center or nearest individual.
z=np.array([0,0,0,1,1,1])
sample_s=silhouette_samples(x[:,None],z)
np.testing.assert_allclose(sample_s,[5/6,7/8,11/14,11/14,7/8,5/6])
single_s=silhouette_samples([[0],[1],[10]],[0,0,1])
assert single_s[-1]==0
for invalid in [np.zeros(6),np.arange(6)]:
    try: silhouette_score(x[:,None],invalid)
    except ValueError: pass
    else: raise AssertionError("Invalid label count should fail")
np.testing.assert_allclose(silhouette_score(x[:,None],[0,0,0,-1,-1,-1]),silhouette_score(x[:,None],z))

# Pair counting, chance adjustment, label permutation, and singleton information.
U=np.array([0,0,1,1]); V=np.array([1,1,0,0]); W=np.array([0,1,0,1])
assert accuracy_score(U,V)==0 and adjusted_rand_score(U,V)==1
cont=contingency_matrix(U,W)
np.testing.assert_array_equal(cont,np.ones((2,2),dtype=int))
I=np.sum(cont*(cont-1)/2)
rows=cont.sum(axis=1); cols=cont.sum(axis=0)
AA=np.sum(rows*(rows-1)/2); BB=np.sum(cols*(cols-1)/2)
expected=AA*BB/6
np.testing.assert_allclose([I,AA,BB,expected],[0,2,2,2/3])
np.testing.assert_allclose((I-expected)/((AA+BB)/2-expected),-.5)
assert adjusted_rand_score(U,W)==-.5
assert adjusted_rand_score([0,0,0],[5,5,5])==1
assert adjusted_rand_score([0,1,2],[3,4,5])==1
r,c=linear_sum_assignment(contingency_matrix(U,V),maximize=True)
assert contingency_matrix(U,V)[r,c].sum()==4
singleton=np.arange(4)
np.testing.assert_allclose([homogeneity_score(U,singleton),completeness_score(U,singleton),v_measure_score(U,singleton)],[1,.5,2/3])
np.testing.assert_allclose(mutual_info_score(U,singleton),np.log(2))
chance_mi=np.mean([mutual_info_score(U,np.array(p)) for p in permutations(singleton)])
np.testing.assert_allclose(chance_mi,np.log(2))
np.testing.assert_allclose(adjusted_mutual_info_score(U,singleton),0,atol=1e-14)
np.testing.assert_allclose(adjusted_mutual_info_score(U,V),1)
assert adjusted_mutual_info_score(U,W)<0
fixed_size_mi=np.mean([mutual_info_score(U,np.array(p)) for p in permutations(W)])
np.testing.assert_allclose(fixed_size_mi,np.log(2)/3)
manual_ami=(mutual_info_score(U,W)-fixed_size_mi)/(np.log(2)-fixed_size_mi)
np.testing.assert_allclose(manual_ami,adjusted_mutual_info_score(U,W))
first={"A":0,"B":0,"C":1,"D":1}; second={"C":3,"A":8,"E":8}
common=sorted(first.keys()&second.keys())
assert common==["A","C"]
assert [first[i] for i in common]==[0,1] and [second[i] for i in common]==[8,3]

# Third published block: observe preprocessing and center-fitting boundaries.
scaler_fits=[]; kmeans_fits=[]
original_scaler=StandardScaler.fit; original_kmeans=KMeans.fit
def capture_scaler(self,X,y=None,sample_weight=None):
    scaler_fits.append(np.array(X,copy=True))
    return original_scaler(self,X,y,sample_weight=sample_weight)
def capture_kmeans(self,X,y=None,sample_weight=None):
    kmeans_fits.append(np.array(X,copy=True))
    return original_kmeans(self,X,y,sample_weight=sample_weight)
StandardScaler.fit=capture_scaler; KMeans.fit=capture_kmeans
try:
    exec(compile(blocks[2],str(CHAPTER)+":block3","exec"),scope)
finally:
    StandardScaler.fit=original_scaler; KMeans.fit=original_kmeans
assert len(scaler_fits)==3 and len(kmeans_fits)==4
for fold,(train,test) in enumerate(KFold(3,shuffle=True,random_state=0).split(scope["features"])):
    expected_train=scope["features"][train]
    np.testing.assert_allclose(scaler_fits[fold],expected_train)
    fitted_rows={tuple(row) for row in scaler_fits[fold]}
    assert not fitted_rows & {tuple(row) for row in scope["features"][test]}
    np.testing.assert_allclose(kmeans_fits[fold+1],StandardScaler().fit_transform(expected_train))
np.testing.assert_allclose(scope["km"].transform([[0],[5],[10]]),[[1,9],[4,4],[9,1]])
query=np.array([0,1,5,9,10])[:,None]
dist_features=scope["km"].transform(query)
np.testing.assert_allclose(dist_features@np.array([1,-1]),[-8,-8,0,8,8])
coef=np.linalg.lstsq(np.c_[query,np.ones(5)],dist_features@np.array([1,-1]),rcond=None)[0]
assert np.sum((np.c_[query,np.ones(5)]@coef-dist_features@np.array([1,-1]))**2)>0
print("PASS: 3 published blocks; Lloyd/local optimum/seeding/scaling;"
      " unit observations versus normalized centers; DBSCAN boundaries;"
      " reachability and MST; EOM; silhouette/ARI/AMI; all CV scaler and center fits")
print(f"Chapter SHA-256: {hashlib.sha256(CHAPTER.read_bytes()).hexdigest()}")
