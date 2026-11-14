# TinyBERT implementation fixtures

Unmodified upstream files from Huawei Noah's Ark Lab:
https://github.com/huawei-noah/Pretrained-Language-Model/tree/0598f02d7fc4eaa7b4cbc1e9d7ab18bc875c24f9/TinyBERT

- task_distill.py: SHA-256: 21c7b70d270cefb50ec05f116554f7af6df3df516d344414c5caafd3142618a3.
- modeling.py: SHA-256: 16a7947c21bb72b51458b54953227fa4a404fbc849d8b9f6b158359456307bb2.

Both retain their upstream Apache-2.0 headers. The verification tool extracts
only the intermediate-loss AST body and the soft_cross_entropy function;
it does not import or run the training script, model downloads or logger.
The student-class AST is inspected for its shared fit_dense projection.

acl-copyright.html is a byte-preserved snapshot of
https://aclanthology.org/faq/copyright/, retrieved 2026-11-05.
SHA-256: d27a4182ffd579b137abf42dab46c7cb836610a616196395dd66b60284b2c237.
It supplies the separate rights basis for the paper illustrations.
