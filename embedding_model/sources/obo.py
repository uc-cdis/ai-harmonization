"""The OBO Foundry ontologies: terms, synonyms and definitions.

Ontologies reuse each other's terms, so only each ontology's own terms are
read. The file read for each:

- MONDO, HP, Uberon and OBA: the base release, which leaves out the terms
  borrowed from other ontologies, at the date that the manifest pins;
- DUO: its main release, at the date that the manifest pins;
- CMO and MMO: their main file. No dated file can be downloaded, so their
  current file is kept after the first fetch, as for a service.

Each ontology's file states its license.
"""

import re

import pandas as pd

from embedding_model.download import downloaded, file_sha256, modified
from embedding_model.sources.common import pinned
from embedding_model.sources.owl import OWL, RDFS, grouped, iri_of, read_owl

PURL = "http://purl.obolibrary.org/obo"
EFO = "http://www.ebi.ac.uk/efo/"
# Each ontology's name, its file at a dated release (None when it publishes
# none), and its file without a date.
ONTOLOGIES = {
    "mondo": ("MONDO", "{purl}/mondo/releases/{date}/mondo-base.owl", None),
    "hp": ("HP", "{purl}/hp/releases/{date}/hp-base.owl", None),
    "uberon": ("UBERON", "{purl}/uberon/releases/{date}/uberon-base.owl", None),
    "oba": ("OBA", "{purl}/oba/releases/{date}/oba-base.owl", None),
    "cmo": ("CMO", None, "{purl}/cmo.owl"),
    "mmo": ("MMO", None, "{purl}/mmo.owl"),
    "duo": ("DUO", "{purl}/duo/releases/{date}/duo.owl", None),
}
OBO_IN_OWL = "{http://www.geneontology.org/formats/oboInOwl#}"
DEFINITION = "{" + PURL + "/}IAO_0000115"
REPLACED_BY = "{" + PURL + "/}IAO_0100001"
CONSIDER = OBO_IN_OWL + "consider"
SYNONYM_TYPE = OBO_IN_OWL + "hasSynonymType"
SCOPES = {
    OBO_IN_OWL + f"has{scope.title()}Synonym": scope
    for scope in ("EXACT", "BROAD", "NARROW", "RELATED")
}
# The properties that an ontology's header can state its license with.
LICENSES = ("{http://purl.org/dc/terms/}license", "{" + PURL + "/}terms_license")
# A class is obsolete when deprecated, or when it is a subclass of this.
OBSOLETE_CLASS = "http://www.geneontology.org/formats/oboInOwl#ObsoleteClass"
# Ontologies mark an obsolete term by putting "obsolete" before its label, as
# in "obsolete hearing loss". The word is stripped, even when written twice or
# as "obsoleted".
OBSOLETE_PREFIX = re.compile(r"^(obsoleted?\s+)+", re.IGNORECASE)
VERSION_IRI = OWL + "versionIRI"
RELEASED = re.compile(r"/releases/(\d{4}-\d{2}-\d{2})/")
CITATION = "embedding_model/docs/sources.md#ontologies"


def release_name(ontology):
    """The manifest's name for an ontology's release, such as ``HP release``."""
    return f"{ONTOLOGIES[ontology][0]} release"


def short_id(iri):
    """An IRI as the short id OBO files write, such as ``HP:0004322``.

    A synonym type defined inside one ontology, such as
    ``http://purl.obolibrary.org/obo/hp#layperson``, keeps only its name.

    Args:
        iri (str): The IRI.

    Returns:
        str: The short id, or the IRI when it is in no known namespace.
    """
    if "#" in iri:
        return iri.rsplit("#", 1)[1]
    for namespace in (PURL + "/", EFO):
        if iri.startswith(namespace):
            return iri[len(namespace) :].replace("_", ":", 1)
    return iri


def read_ontology(stream, ontology):
    """An ontology's own terms, with their synonyms and definitions.

    Only the ontology's own terms are read: those whose id has its prefix,
    such as ``HP:``. A term needs a name, its ``rdfs:label``, such as "Short
    stature", unless it is obsolete. An obsolete term is kept with what
    replaced it, even without a name. A term can have several definitions,
    and a synonym can have several types. All are kept.

    Args:
        stream (file): The OWL file, opened in binary mode.
        ontology (str): The ontology's key in ONTOLOGIES, such as ``hp``.

    Returns:
        tuple[dict[str, pd.DataFrame], dict]: The tables ``terms`` (id, label,
            whether obsolete, ``replaced_by`` and ``consider``), ``synonyms``
            (with scope and types) and ``definitions``, each with an
            ``ontology`` column; and what the file's header states: its
            ``license`` and the ``release`` date in its version IRI, each
            ``""`` when not stated.
    """
    prefix = ONTOLOGIES[ontology][0] + ":"
    found = read_owl(stream)
    header = grouped(found["ontology"])
    license_ = next((header[key][0].strip() for key in LICENSES if header.get(key)), "")
    released = RELEASED.search((header.get(VERSION_IRI) or [""])[0])
    stated = {"license": license_, "release": released.group(1) if released else ""}
    rows = {"terms": [], "synonyms": [], "definitions": []}
    for iri, annotations in found["classes"].items():
        values = grouped(annotations)
        term = short_id(iri)
        if not term.startswith(prefix):
            continue
        label = (values.get(RDFS + "label") or [""])[0].strip()
        obsolete = "true" in values.get(OWL + "deprecated", []) or (
            OBSOLETE_CLASS in values.get(RDFS + "subClassOf", [])
        )
        if not label and not obsolete:
            continue
        if obsolete:
            label = OBSOLETE_PREFIX.sub("", label)
        rows["terms"].append(
            {
                "id": term,
                "ontology": ontology,
                "label": label,
                "obsolete": obsolete,
                "replaced_by": "|".join(
                    short_id(v) for v in values.get(REPLACED_BY, [])
                ),
                "consider": "|".join(short_id(v) for v in values.get(CONSIDER, [])),
                "open": True,
            }
        )
        rows["definitions"] += [
            {"id": term, "ontology": ontology, "definition": text.strip()}
            for text in values.get(DEFINITION, [])
        ]
        for prop, scope in SCOPES.items():
            for text in values.get(prop, []):
                axioms = found["axioms"].get((iri, iri_of(prop), text), [])
                kinds = sorted(
                    {short_id(k) for a in axioms for k in a.get(SYNONYM_TYPE, [])}
                )
                rows["synonyms"].append(
                    {
                        "id": term,
                        "ontology": ontology,
                        "name": text.strip(),
                        "scope": scope,
                        "synonym_types": "|".join(kinds),
                    }
                )
    return {
        table: pd.DataFrame(found_rows) for table, found_rows in rows.items()
    }, stated


def fetch_source(downloads, manifest):
    """Fetch every ontology at the release that the manifest pins.

    Args:
        downloads (Path): Where downloaded files are kept between fetches.
        manifest (dict): The committed manifest; each ontology's release entry
            gives its date. CMO's and MMO's give none: their current file is
            kept.

    Returns:
        tuple[dict[str, pd.DataFrame], list[dict]]: The tables, every
            ontology's rows together, and the manifest's sources.
    """
    tables = {"terms": [], "synonyms": [], "definitions": []}
    sources = []
    for ontology, (_, dated, undated) in ONTOLOGIES.items():
        pin = pinned(manifest, release_name(ontology))
        if dated:
            url = dated.format(purl=PURL, date=pin["version"])
            path = downloaded(url, downloads / f"{ontology}-{pin['version']}.owl")
        else:
            url = undated.format(purl=PURL)
            path = downloaded(url, downloads / f"{ontology}.owl")
        with open(path, "rb") as stream:
            read, stated = read_ontology(stream, ontology)
        for table, frame in read.items():
            tables[table].append(frame)
        print(f"  {ontology:<7}{len(read['terms']):>8,} terms")
        if not stated["license"]:
            print(f"  {ontology} states no license")
        sources.append(
            {
                "role": "release" if dated else "service",
                "name": release_name(ontology),
                "location": url,
                "version": pin["version"] if dated else stated["release"],
                **({} if dated else {"fetched": modified(path)}),
                "sha256": file_sha256(path),
                "license": stated["license"],
                "citation": CITATION,
            }
        )
    keys = {
        "terms": ["id"],
        "synonyms": ["id", "scope", "name", "synonym_types"],
        "definitions": ["id", "definition"],
    }
    return {
        table: pd.concat(frames, ignore_index=True).sort_values(
            keys[table], ignore_index=True
        )
        for table, frames in tables.items()
    }, sources
