# MovieLens Recommendation System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project showcases the development of a sophisticated recommendation system for a media streaming platform, inspired by Netflix, using the MovieLens 1B Synthetic Dataset. The system employs a hybrid approach, integrating collaborative filtering, content-based filtering, and graph-based recommendations to deliver personalised movie suggestions.

## Project Organisation

```
├── LICENSE     <- Open-source license if one is chosen
│
├── Makefile    <- Makefile with convenience commands like `make data` or `make train`
│
├── README.md   <- The top-level README for developers using this project.
│
├── data
│   │
│   ├── external    <- Data from third party sources.
│   │
│   ├── interim     <- Intermediate data that has been transformed.
│   │
│   ├── processed   <- The final, canonical data sets for modeling.
│   │
│   └── raw         <- The original, immutable data dump.
│
├── environment.yml <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── models          <- Trained and serialsed models, model predictions, or model summaries
│
├── notebooks       <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
└── src     <- Source code for use in this project.
    │ 
    ├── __init__.py         <- Makes src a Python module
    │
    ├── dataset.py          <- Scripts to download or generate data
    │
    ├── features.py         <- Code to create features for modeling
    │
    ├── modeling
    │   │
    │   ├── __init__.py
    │   │
    │   ├── predict.py      <- Code to run model inference with trained models
    │   │
    │   └── train.py        <- Code to train models
    │
    └── plots.py            <- Code to create visualisations


```

--------

