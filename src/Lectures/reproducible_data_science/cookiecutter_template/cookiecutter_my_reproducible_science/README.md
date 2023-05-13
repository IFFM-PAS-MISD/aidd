Reproducible Science
====================

A boilerplate for reproducible and transparent science with close resemblances to the philosophy of [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science): *A logical, reasonably standardized, but flexible project structure for doing and sharing data science work.*

Requirements
------------
Install `cookiecutter` command line: `pip install cookiecutter`
or:
`conda install cookiecutter`

Usage
-----
To start a new science project:

`cookiecutter ./cookiecutter-my-reproducible-science`

Project Structure
-----------------


```
    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── bin				<- Your compiled model code can be stored here (not tracked by git)  
    ├── data
    │   ├── processed			<- Processed data
    │   └── raw			<- The original, immutable data dump.
    ├── reports			<- For a manuscript source, e.g., LaTeX, Beamer, Markdown, etc.
    │   ├── conference_papers		
    │   ├── figures			<- Figures for the manuscripts and presentations
    │   ├── journal_papers
    │   ├── presentations
    └── src
        ├── data_processing		<- Source code data processing
        ├── models			<- Source code for your own model
        ├── tools			<- Any helper scripts go here
        └── visualization		<- Scripts for visualisation of your results, e.g., matlab, matplotlib, ggplot2 related.
```

License
-------
This project is licensed under the terms of the [BSD License](/LICENSE)
