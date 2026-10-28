# Licenses and references

The benchmarks and the training sources are built from public data that others
publish. This page gives the license or terms of each one, and the reference to
cite. It also says how single restricted items are left out, and what this
repository shares.

## What this repository shares

The repository shares no data. Each benchmark build and each source fetch
downloads the originals from their publishers. It makes its files on your
machine, under `embedding_model/inputs/`. The pairs are made there too.

The manifests are committed. They record each input's version, checksum,
license and citation, but they hold no data (see [manifests.md](manifests.md)).

For the GDC benchmark, building on your machine is required: see [GDC](#gdc).

## Benchmarks

### TOPMed

**Website:**
[UW-GAC/topmed-dcc-harmonized-phenotypes](https://github.com/UW-GAC/topmed-dcc-harmonized-phenotypes).

**License:** The documentation repository states no license.

**Reference:**

> Stilp, A. M., Emery, L. S., Broome, J. G., Buth, E. J., Khan, A. T.,
> Laurie, C. A., Wang, F. F., Wong, Q., Chen, D., D'Augustine, C. M.,
> Heard-Costa, N. L., Hohensee, C. R., Johnson, W. C., Juarez, L. D., Liu, J.,
> Mutalik, K. M., Raffield, L. M., Wiggins, K. L., de Vries, P. S., et al.
> (2021). A System for Phenotype Harmonization in the National Heart, Lung,
> and Blood Institute Trans-Omics for Precision Medicine (TOPMed) Program.
> *American Journal of Epidemiology*, *190*(10), 1977–1992.
> https://doi.org/10.1093/aje/kwab115

### GDC

**Website:** GDC-SM's [Zenodo record](https://doi.org/10.5281/zenodo.14963588),
the [GDC Data Dictionary](https://docs.gdc.cancer.gov/Data_Dictionary/), and
Magneto's code,
[VIDA-NYU/magneto-matcher](https://github.com/VIDA-NYU/magneto-matcher).

**License:** The ground truth is [CC BY
4.0](https://creativecommons.org/licenses/by/4.0/), and the GDC dictionary is
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). Seven of the ten
papers are published under [CC BY-NC-ND
4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), a Creative Commons
NoDerivatives license. It allows sharing a file as published, but not a
modified version of it. A benchmark table is a modified version: it is
repaired and joined to the expert mappings. So GDC-SM distributes only the
ground truth, with instructions for rebuilding the tables from the publishers'
originals. Likewise, this repository distributes no benchmark files: the
builder downloads the originals and makes the tables on your machine.

**References:**

> Santos, A., Wu, E., Lopez, R., Keegan, S., Pena, E., Liu, W., Liu, Y.,
> Fenyo, D., & Freire, J. (2025). *GDC-SM: The GDC Schema Matching Benchmark*
> (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.14963588

> Liu, Y., Pena, E. H. M., Santos, A., Wu, E., & Freire, J. (2025). Magneto:
> Combining Small and Large Language Models for Schema Matching. *Proceedings
> of the VLDB Endowment*, *18*(8), 2681–2694.
> https://doi.org/10.14778/3742728.3742757

Each paper's table, with its SHA-256, and its article's DOI are listed in
`embedding_model/benchmarks/manifests/gdc.json`.

## Training sources

Each source's license was checked. Every source is free to use, under the
license or terms below.

### NCIt

**Website:** [evs.nci.nih.gov](https://evs.nci.nih.gov/).

**License:** NCIt is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as EVS's
[Terms of Use](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ThesaurusTermsofUse.htm)
state.

**Reference:**

> Sioutos, N., Coronado, S. d., Haber, M. W., Hartel, F. W., Shaiu, W.-L., &
> Wright, L. W. (2007). NCI Thesaurus: A semantic model integrating
> cancer-related clinical and molecular information. *Journal of Biomedical
> Informatics*, *40*(1), 30–43. https://doi.org/10.1016/j.jbi.2006.02.013

### Ontologies

**Website:** [OBO Foundry](https://obofoundry.org/), which lists all seven. The
table links each one's page there.

**License:** Each ontology's file states its license, and the table below
shows it. For CMO and MMO, the OBO Foundry pages give [CC0
1.0](https://creativecommons.org/publicdomain/zero/1.0/), while the files
state [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| ontology | website | license |
|---|---|---|
| MONDO, diseases | [mondo](https://obofoundry.org/ontology/mondo.html) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| HP, phenotypes | [hp](https://obofoundry.org/ontology/hp.html) | [HPO license](https://human-phenotype-ontology.github.io/license.html) |
| Uberon, anatomy | [uberon](https://obofoundry.org/ontology/uberon.html) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| OBA, biological attributes | [oba](https://obofoundry.org/ontology/oba.html) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| CMO, clinical measurements | [cmo](https://obofoundry.org/ontology/cmo.html) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| MMO, measurement methods | [mmo](https://obofoundry.org/ontology/mmo.html) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| DUO, data use | [duo](https://obofoundry.org/ontology/duo.html) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The HPO license names no standard license. It asks users to:

- acknowledge and cite the HPO Consortium;
- show the date or version of each HPO file that they display;
- leave the files and their logical relationships unaltered;
- say in a service or product: "This service/product is using the Human
  Phenotype Ontology (_version information_)."

The HP file's header names `https://hpo.jax.org/app/license`, which no
longer loads. The license is now at the address in the table.

**References:**

> Vasilevsky, N. A., Matentzoglu, N. A., Toro, S., Flack, J. E., Hegde, H.,
> Unni, D. R., Alyea, G. F., Amberger, J. S., Babb, L., Balhoff, J. P.,
> Bingaman, T. I., Burns, G. A., Buske, O. J., Callahan, T. J., Carmody, L.
> C., Cordo, P. C., Chan, L. E., Chang, G. S., Christiaens, S. L., et al.
> (2022). Mondo: Unifying diseases for the world, by the world.
> https://doi.org/10.1101/2022.04.13.22273750

> Gargano, M. A., Matentzoglu, N., Coleman, B., Addo-Lartey, E. B.,
> Anagnostopoulos, A., Anderton, J., Avillach, P., Bagley, A. M., Bakštein,
> E., Balhoff, J. P., Baynam, G., Bello, S. M., Berk, M., Bertram, H., Bishop,
> S., Blau, H., Bodenstein, D. F., Botas, P., Boztug, K., et al. (2023). The
> Human Phenotype Ontology in 2024: phenotypes around the world. *Nucleic
> Acids Research*, *52*(D1), D1333–D1346. https://doi.org/10.1093/nar/gkad1005

> Mungall, C. J., Torniai, C., Gkoutos, G. V., Lewis, S. E., & Haendel, M. A.
> (2012). Uberon, an integrative multi-species anatomy ontology. *Genome
> Biology*, *13*(1). https://doi.org/10.1186/gb-2012-13-1-r5

> Stefancsik, R., Balhoff, J. P., Balk, M. A., Ball, R. L., Bello, S. M.,
> Caron, A. R., Chesler, E. J., de Souza, V., Gehrke, S., Haendel, M., Harris,
> L. W., Harris, N. L., Ibrahim, A., Koehler, S., Matentzoglu, N., McMurry, J.
> A., Mungall, C. J., Munoz-Torres, M. C., Putman, T., et al. (2023). The
> Ontology of Biological Attributes (OBA)—computational traits for the life
> sciences. *Mammalian Genome*, *34*(3), 364–378.
> https://doi.org/10.1007/s00335-023-09992-1

> Smith, J. R., Park, C. A., Nigam, R., Laulederkind, S. J., Hayman, G. T.,
> Wang, S.-J., Lowry, T. F., Petri, V., Pons, J. D., Tutaj, M., Liu, W.,
> Worthey, E. A., Shimoyama, M., & Dwinell, M. R. (2013). The clinical
> measurement, measurement method and experimental condition ontologies:
> expansion, improvements and new applications. *Journal of Biomedical
> Semantics*, *4*(1). https://doi.org/10.1186/2041-1480-4-26

> Lawson, J., Cabili, M. N., Kerry, G., Boughtwood, T., Thorogood, A., Alper,
> P., Bowers, S. R., Boyles, R. R., Brookes, A. J., Brush, M., Burdett, T.,
> Clissold, H., Donnelly, S., Dyke, S. O. M., Freeberg, M. A., Haendel, M. A.,
> Hata, C., Holub, P., Jeanson, F., et al. (2021). The Data Use Ontology to
> streamline responsible access to human biomedical datasets. *Cell
> Genomics*, *1*(2), 100028. https://doi.org/10.1016/j.xgen.2021.100028

### LOINC

**Website:** [loinc.org](https://loinc.org/).

**License:** The [LOINC License](https://loinc.org/license/): free, with
attribution. Content that belongs to a third party carries a notice, and is not
open (see [Restricted items](#restricted-items)).

**Reference:**

> McDonald, C. J., Huff, S. M., Suico, J. G., Hill, G., Leavelle, D., Aller,
> R., Forrey, A., Mercer, K., DeMoor, G., Hook, J., Williams, W., Case, J.,
> Maloney, P., & Laboratory LOINC Developers. (2003). LOINC, a Universal
> Standard for Identifying Laboratory Observations: A 5-Year Update.
> *Clinical Chemistry*, *49*(4), 624–633. https://doi.org/10.1373/49.4.624

### CDISC

**Website:**
[CDISC Controlled Terminology](https://www.cdisc.org/standards/terminology/controlled-terminology).

**License:** NCI states that CDISC Terminology can be used "for free, without
licensing restrictions", on its
[CDISC terminology page](https://www.cancer.gov/about-nci/organization/cbiit/data-standards/cdisc).

**Reference:**

> CDISC. (n.d.). *Controlled Terminology*. Retrieved October 22, 2026, from
> https://www.cdisc.org/standards/terminology/controlled-terminology

### NIH CDE

**Website:** [cde.nlm.nih.gov](https://cde.nlm.nih.gov/).

**License:** NLM's [web policies](https://www.nlm.nih.gov/web_policies.html)
state that U.S. government works on its sites "may be freely used or reproduced
without permission in the U.S.". Content contributed by others may be
copyrighted. Each element's `copyrightStatus` says which: only "Public domain,
free to use" is open.

**Reference:**

> National Library of Medicine. (n.d.). *NIH Common Data Elements (CDE)
> Repository*. Retrieved October 22, 2026, from https://cde.nlm.nih.gov/

### caDSR

**Website:** [cadsr.cancer.gov](https://cadsr.cancer.gov/).

**License:** NCI states that "unless otherwise indicated, all text within
National Cancer Institute (NCI) products is free of copyright"
([Reuse of NCI Information](https://www.cancer.gov/policies/copyright-reuse)).
An element of a proprietary measure has a Proprietary Measure Permission
document, and is not open.

**Reference:**

> Covitz, P. A., Hartel, F., Schaefer, C., De Coronado, S., Fragoso, G., Sahni,
> H., Gustafson, S., & Buetow, K. H. (2003). caCORE: A common infrastructure
> for cancer informatics. *Bioinformatics*, *19*(18), 2404–2412.
> https://doi.org/10.1093/bioinformatics/btg335

### NHANES

**Website:** [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/).

**License:** CDC states that most of the information on its websites is in the
public domain. It "may be freely used or reproduced without obtaining copyright
permission"
([Use of Agency Materials](https://www.cdc.gov/other/agencymaterials.html)).

**Reference:**

> Centers for Disease Control and Prevention, National Center for Health
> Statistics. (n.d.). *National Health and Nutrition Examination Survey*.
> Retrieved October 22, 2026, from https://wwwn.cdc.gov/nchs/nhanes/

### PhenX

**Website:** [phenxtoolkit.org](https://www.phenxtoolkit.org/).

**License:** PhenX's [FAQ](https://www.phenxtoolkit.org/help/faq) states that
most of its protocols are freely available, and a small share have limited
availability. Each protocol's `availability` says which: only "Available" is
open.

**Reference:**

> Hamilton, C. M., Strader, L. C., Pratt, J. G., Maiese, D., Hendershot, T.,
> Kwok, R. K., Hammond, J. A., Huggins, W., Jackman, D., Pan, H., Nettles, D.
> S., Beaty, T. H., Farrer, L. A., Kraft, P., Marazita, M. L., Ordovas, J. M.,
> Pato, C. N., Spitz, M. R., Wagener, D., et al. (2011). The PhenX Toolkit:
> Get the Most From Your Measures. *American Journal of Epidemiology*,
> *174*(3), 253–260. https://doi.org/10.1093/aje/kwr193

## Restricted items

Some sources also mark single items as restricted, in a field of their own.
For example, LOINC marks a code whose content belongs to a third party. The
fetch reads these restriction fields too. Each source's main table has an
`open` column: true when the item's restriction field allows its use.
`pairs prepare` uses only open items (see
[pairs.md](pairs.md#only-open-items)).

Where the field holds one of a list of values, an item is open only when its
value is known to allow use. A value that the fetch has not seen before means
the item is not open. The value is printed, so it can be reviewed. For caDSR,
the field is a document instead. A new kind of document is printed too, but
only a Proprietary Measure Permission closes an element.

| source | restriction field | open when |
|---|---|---|
| NCIt, ontologies, CDISC | none per item | always: the source's license covers every item |
| LOINC | `EXTERNAL_COPYRIGHT_NOTICE`, and a form row's `AdditionalCopyright` | they are empty, on the item and on the panels that it sits in. With closed panels left out, a code is open when it sits in an open panel, or in no panel and has no notice. A notice marks third-party content. |
| NIH CDE | `copyrightStatus`, `noRenderAllowed` | `copyrightStatus` is "Public domain, free to use", and `noRenderAllowed` is not true |
| caDSR | a "Proprietary Measure Permission" reference document | the element has none |
| NHANES | "Use Constraints", per data file | it is "None" or "RDC Only". "RDC Only" restricts access to the data, not the documentation read here. |
| PhenX | the protocol's availability | it is "Available" |
