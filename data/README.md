# Datasets

`reproduce.py` uses the **LMA Effort** dataset (Kim et al., ACM TAP 2022). Place
the dataset zip in this directory as `extracted_fingerless-*.zip`, or an already
extracted `data/extracted_fingerless/` folder of BVH files. On the first run,
`reproduce.py` extracts the zip and builds the dataset cache `lma_effort.pkl`
here. These files are not tracked in git (see `../.gitignore`).

Datasets used in the paper and where to obtain them:

- **LMA Effort Dataset** — Kim et al., ACM TAP 2022. https://doi.org/10.1145/3473041
- **Bandai-Namco Research Motion Dataset 2** — Kobayashi et al., 2023.
  https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset
- **Folk Dance Motion Capture** — Aristidou et al., JOCCH 2015.
  http://dancedb.cs.ucy.ac.cy/

Each dataset keeps its own license; see the respective source for terms.
