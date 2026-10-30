# Efficient DL Systems: error feedback

Copyright (c) 2021 course authors. MIT License.

Original full slide 58 of Week 3, Efficient Deep Learning Systems,
HSE University and Yandex School of Data Analysis, revision
e632aa89ca9e6638d52e1b686095e7442faffbb0.

The source PDF, original README with the staff list, and full MIT license
are retained in `05 Источники/Originals/ddp-2026-10-23/`.
The same license is public at `/licenses/edls-e632aa89-MIT.txt`.

The PNG is a complete page render at 1920 x 1081, without a crop,
translated label, redrawing or removed component. Reproduction command:

```sh
pdftoppm -f 58 -l 58 -scale-to 1920 -png edls-week03.pdf error-feedback
```

The slide shows the first compression with zero initial error, then
the error entering the next gradient. The right-hand C(g) is abbreviated
notation for compression of its g+e input; the textbook caption explains
this without editing the original. The right-edge continuation belongs
to the original page and is not a website crop.

The paper-derived illustrations on slides 60-61 are not copied under the
course license. The PowerSGD mechanism is checked against Algorithm 1
of the paper and illustrated with a separate worked numerical example.
