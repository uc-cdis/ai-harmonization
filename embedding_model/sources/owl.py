"""Read an ontology's annotations from its OWL file, in RDF/XML.

The OWL API writes NCIt's and the OBO Foundry ontologies' files. Such a file
holds one ``owl:Class`` block per term and one ``owl:Axiom`` block per
annotated statement. An axiom adds details to one of the class's statements: a
synonym's type, say, or a definition's source.
"""

import xml.etree.ElementTree as ET

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"
KINDS = {OWL + "Class": "classes", OWL + "AnnotationProperty": "properties"}
# A defined class names its parents inside its definition, as the named
# members of an intersection. This statement stands for each of them.
DEFINED_PARENT = OWL + "intersectionOf"


def statements(element):
    """An element's annotations, as (property, value) tuples.

    A value is the child's text, or the IRI that it points to. The text is kept
    exactly as written, since an axiom finds the statement that it annotates by
    that text. A defined class's parents are kept as ``DEFINED_PARENT``
    statements. Other nested structures are skipped.

    Args:
        element (xml.etree.ElementTree.Element): An ``owl:Class`` or another
            top-level element.

    Returns:
        list[tuple[str, str]]: The statements, in the file's order.
    """
    found = []
    for child in element:
        if not len(child):
            found.append((child.tag, child.get(RDF + "resource") or child.text or ""))
        elif child.tag == OWL + "equivalentClass":
            members = child.find(f"{OWL}Class/{OWL}intersectionOf")
            for member in members if members is not None else []:
                if member.tag == RDF + "Description" and member.get(RDF + "about"):
                    found.append((DEFINED_PARENT, member.get(RDF + "about")))
    return found


def read_owl(stream):
    """The ontology header, classes, annotation properties and axioms of a file.

    Args:
        stream (file): The OWL file, opened in binary mode.

    Returns:
        dict: ``ontology``, the header's statements; ``classes`` and
            ``properties``, each by IRI, as lists of statements; and
            ``axioms``, keyed by (subject, property, value), each a list of
            the axioms' annotations, every annotation a list of values. One
            statement can carry several axioms, such as a synonym from two
            sources. One axiom can carry several values, such as a synonym with
            two types.
    """
    found = {"ontology": [], "classes": {}, "properties": {}, "axioms": {}}
    depth, root = 0, None
    for event, element in ET.iterparse(stream, events=("start", "end")):
        if event == "start":
            root = element if root is None else root
            depth += 1
            continue
        depth -= 1
        if depth != 1:
            continue
        about = element.get(RDF + "about")
        if element.tag == OWL + "Ontology":
            found["ontology"].extend(statements(element))
        elif element.tag in KINDS and about:
            found[KINDS[element.tag]].setdefault(about, []).extend(statements(element))
        elif element.tag == OWL + "Axiom":
            fields = grouped(statements(element))
            key = tuple(
                fields.pop(OWL + name, [""])[0]
                for name in ("annotatedSource", "annotatedProperty", "annotatedTarget")
            )
            found["axioms"].setdefault(key, []).append(fields)
        root.clear()
    return found


def grouped(annotations):
    """An element's statements grouped by property, values in the file's order.

    Args:
        annotations (list[tuple[str, str]]): From ``statements``.

    Returns:
        dict[str, list[str]]: The values of each property.
    """
    values = {}
    for prop, value in annotations:
        values.setdefault(prop, []).append(value)
    return values


def iri_of(tag):
    """The IRI of a property, from its tag in ``{namespace}name`` form."""
    return tag[1:].replace("}", "", 1)
