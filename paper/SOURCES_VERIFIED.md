# Reference checks for the revised manuscript

Checked 8 September 2026. The paper's technical description was checked against
the implementation, not inferred from literature. The references below support
background or methodological attribution; none supplies this project's results.

| Topic | Primary source |
|---|---|
| LSTM rainfall–runoff modelling | [Kratzert et al., HESS 2018](https://hess.copernicus.org/articles/22/6005/2018/) |
| Global neural flood prediction | [Nearing et al., Nature 2024](https://doi.org/10.1038/s41586-024-07145-1); [authors' preprint](https://arxiv.org/abs/2307.16104) |
| Transformer architecture | [Vaswani et al., NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) |
| Numerical embeddings | [Gorishniy et al., NeurIPS 2022](https://papers.nips.cc/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf) |
| FiLM | [Perez et al., AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11671) |
| River topology | [Kirschstein and Sun, ICML 2024](https://proceedings.mlr.press/v235/kirschstein24a.html) |
| Focal loss | [Lin et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html) |
| Ensembles | [Lakshminarayanan et al., NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) |
| Temperature scaling | [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) |
| Structured cross-validation | [Roberts et al., Ecography](https://nsojournals.onlinelibrary.wiley.com/doi/abs/10.1111/ecog.02881) |
| Precision–recall evaluation | [Saito and Rehmsmeier, PLOS ONE 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) |
| POWER meteorology | [NASA methodology](https://power.larc.nasa.gov/docs/methodology/meteorology/) |
| GloFAS reanalysis | [Harrigan et al., ESSD 2020](https://essd.copernicus.org/articles/12/2043/2020/) |
| Discharge API | [Open-Meteo Flood API](https://open-meteo.com/en/docs/flood-api) |
| Country outline | [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/) and the maintainer's GeoJSON linked in `data/README.md` |

The Sri Lankan Saubhagya et al. reference is retained from the original paper and
the repository's `docs/SRI_LANKA_LITERATURE.md`. Its publisher URL is
[Forecasting 7(2), 29](https://www.mdpi.com/2571-9394/7/2/29), DOI
10.3390/forecast7020029. The publisher blocked full-text retrieval during this
revision. The revised paper retains only the broad, locally documented method
and location description; it removes the former detailed accuracy and
event-count criticism. Recheck the full paper during the final bibliography
audit.

The original disaster-trend/casualty claims, extended quotations from related
studies, unverified operational stage values and broad statements about focal
loss were removed. The study-area figure does not use administrative polygons
as hydrological basin boundaries.
