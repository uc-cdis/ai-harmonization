"""caDSR: each data element, around its long name.

The source side holds the element's short name, some of its alternate names,
and its question texts. An alternate name is used when its type holds names,
in words or as codes and variable names. Alternate names of other types hold
mostly context names, ids, model paths, table names or mixed text, so they are
not used. A type that the reader has not seen is printed, and its names are not
used. A question text is a reference document whose type ends in "Question
Text", such as the preferred and alternate question texts. caDSR writes a
placeholder sentence when an element has no question text, and this sentence
is not used.

The description is the element's definition, its concepts' definitions joined
by a space. Elements of old software models end their long name with a Java
type, such as ``java.lang.String``, and their definition with a phrase about
that type. Both are removed, since the datatype gives the type. Each value is
its meaning, or the value itself when it has no
meaning, in the element's order. The codes written before the values are
removed as ``common.value_labels`` says. The type is the element's own
datatype, such as ``CHARACTER``.
"""

import re
from collections import defaultdict

from embedding_model.pairs.common import (
    CURRENT,
    DRAFT,
    RETIRED,
    Item,
    Text,
    cadsr_definition,
    value_labels,
    without_java,
)
from embedding_model.sources.common import report_new

STATUSES = {
    "RELEASED": CURRENT,
    "RELEASED-NON-CMPLNT": CURRENT,
    "DRAFT NEW": DRAFT,
    "DRAFT MOD": DRAFT,
    "RETIRED ARCHIVED": RETIRED,
    "RETIRED PHASED OUT": RETIRED,
    "RETIRED WITHDRAWN": RETIRED,
    "RETIRED DELETED": RETIRED,
}
# Reference documents that hold a question's wording, such as "Preferred
# Question Text" or "Application Standard Question Text", end in these words.
QUESTION_TYPE = "Question Text"
# What caDSR writes when an element has no question: "Data Element Abcd Text
# does not have Preferred Question Text."
NO_QUESTION = re.compile(
    r"^Data Element .+ does not have (?:Preferred |Alternate )?Question Text\.?$"
)
# The alternate-name types that hold another name for the element. The name is
# in words, or a variable name or a code, such as the GDC Alt Name
# "progression_or_recurrence" or the CDISC SDTM Variable "TRTESTCD". The other
# types hold mostly context names, ids, model paths, table names or mixed text.
NAME_TYPES = {
    "ACT Name",
    "AU Cancer Dataset",
    "Biomarker Synonym",
    "C3D Name",
    "C3DC Alt Name",
    "C3PR Name",
    "CCDI Alt Name",
    "CCDI-Fed Alt Name",
    "CDISC",
    "CDISC SDTM Variable",
    "CIDC Alt Name",
    "CMB Alt Name",
    "COG Form Element",
    "COG GDE Short Name",
    "Context Short Name",
    "CRDC Alt Name",
    "CRDC Model Alt Name",
    "CSTE",
    "CTDC Alt Name",
    "CTEP CDISC CDE",
    "DCP Alt Name",
    "DICOM Attribute Name",
    "eMERGE Name",
    "GC Alt Name",
    "GDC Alt Name",
    "GDC_Name",
    "HCMI Name",
    "HL7 Name",
    "HTAN Alt Name",
    "IARC Cancer Dataset",
    "ICDC Alt Name",
    "IDC Alt Name",
    "ISA-TAB-Nano Col Nm",
    "LNC",
    "LOINC_NAME",
    "Map:CTGOV",
    "Map:CTR&Rr2",
    "Map:CTRR",
    "Map:PRM",
    "Map:WHO",
    "MATCH Alt Name",
    "MATCHBox Alt Name",
    "MCL Alt Name",
    "MedDRA_LLT_Name",
    "MedDRA_PT_Name",
    "NAACCR Alt Name",
    "NAACCR Item Name",
    "NAACCR VM Name",
    "NCCR Alt Name",
    "NCI_MO_CODE",
    "NIAID Alt Name",
    "NICHD Alt Name",
    "NIH COVID",
    "NINDS Alt Name",
    "NMDP_FN",
    "NRDS Alternate Name",
    "OBIB",
    "OID, ACRIN",
    "OID, Alliance",
    "OID, AMC",
    "OID, CCR",
    "OID, CCTG",
    "OID, COG",
    "OID, CTSU",
    "OID, Mayo",
    "OID, NCIP",
    "OID, NHC",
    "OID, NRDS Standard",
    "OID, NRG",
    "OID, RTOG",
    "OID, SWOG",
    "OID, Theradex",
    "OMOP Name",
    "PCDC Alt Name",
    "PCORNet Name",
    "PDC Alt Name",
    "Ped Cancer Alt Name",
    "PMI Alt Name",
    "Prior Preferred Name",
    "PSDC Alt Name",
    "QIN Breast Study",
    "RQRS Column Name",
    "SAE Rave Integration",
    "Screen Label",
    "SDO Alt Name",
    "Sentinel Name",
    "Synonym",
    "TARGET Alt Name",
    "TCGA_Name",
    "TCGA_XML_Name",
    "TCIA Alt Name",
    "TW Cancer Dataset",
    "UK Cancer Dataset",
    "VM Alt Name",
    "XML Tag",
}
# The alternate-name types that are not used as names: contexts, ids, model
# paths, table names and mixed text. Examples are the USED_BY context "ABTC",
# the UML Class:UML Attr "Patient:race", and "Measure Alt Name", which mixes
# score names with ids. A type in neither list is new.
OTHER_TYPES = {
    "3.1 VM Name",
    "ABBREVIATION",
    "ACRIN Alt Name",
    "ACRIN Question No.",
    "ACT Table",
    "AHRQ Identifier",
    "AU data field name",
    "C3DC Table",
    "CCDI Table",
    "CDISC SDTM Domain",
    "CIDC Table",
    "COG GDE #",
    "CONTEXT NAME",
    "CRITERION_NAME",
    "CTDC Table",
    "DICOM Type",
    "DICOM_TAG",
    "ePRO Field Name",
    "FHIR US CORE Name",
    "FHIR US CORE Table",
    "GC Table",
    "GME_XMLLocReference",
    "GO_CODE",
    "HCT_BRIDG",
    "HCT_BRIDG_5",
    "HISTORICAL_CDE_ID",
    "IDC Table",
    "LOINC_CODE",
    "Manually-curated",
    "Map:AE",
    "Map:AIM v4 rv48",
    "Map:BRIDGSCC",
    "Map:BRIDGv2.2",
    "Map:BRIDGv5.3",
    "Map:C3PR",
    "Map:C3PRv2.9",
    "Map:caAERSv2.2",
    "Map:CDASHv1.1",
    "Map:CDISCLabv1.0.1",
    "Map:CDMHv1.0",
    "Map:CTOM",
    "Map:CTRPv1.0",
    "Map:CTRPv3.8",
    "Map:CTRRr3",
    "Map:CTRv1.0",
    "Map:DICOM",
    "Map:HCTv1.0",
    "Map:HL7SD",
    "Map:HL7SDr1",
    "Map:HL7SP",
    "Map:HSDBv1.0",
    "Map:i2b2/ACTv1.4",
    "Map:ICSRr2",
    "Map:Lab",
    "Map:LabViewer2.2",
    "Map:LSDAMv2.2.3Plus",
    "Map:NCI CRF Standard",
    "Map:OMOPv5.2",
    "Map:PCORNetv3.1",
    "Map:PCORNetv4.0",
    "Map:PSC",
    "Map:PSCv2.6",
    "Map:SDTM IGv3.1.1",
    "Map:SDTM IGv3.1.2",
    "Map:SDTM IGv3.1.3",
    "Map:SDTM IGv3.2",
    "Map:SEER 2015",
    "Map:Sentinelv6.0.2",
    "Map:Statistics v1.0",
    "Map:TDM",
    "Map:TDMv2",
    "Measure Alt Name",
    "MedDRA_LLT_Code",
    "MedDRA_PT_Code",
    "NAACCR Item #",
    "OBIB Identifier",
    "OMOP Table",
    "PCORnet Table",
    "PSDC Table",
    "Sentinel Table",
    "System-Generated",
    "TW data field name",
    "UML Class:UML Attr",
    "UML Qualified Attr",
    "UML Qualified Class",
    "USED_BY",
}


def items(tables):
    """caDSR's data elements as items.

    Args:
        tables (dict[str, pd.DataFrame]): The tables ``sources fetch`` wrote.

    Returns:
        list[Item]: One item per element.
    """
    elements = tables["elements"]
    report_new(
        elements.workflow_status, set(STATUSES), "WORKFLOWSTATUS", "read as current"
    )
    texts = defaultdict(list)
    alternate = tables["alternate_names"]
    report_new(alternate.type, NAME_TYPES | OTHER_TYPES, "type", "not used")
    for row in alternate[alternate.type.isin(NAME_TYPES)].itertuples():
        texts[row.public_id].append(Text(row.name, row.type))
    documents = tables["documents"]
    questions = documents[documents.type.str.endswith(QUESTION_TYPE)]
    for row in questions[~questions.text.str.match(NO_QUESTION)].itertuples():
        texts[row.public_id].append(Text(row.text, row.type, question=True))
    values = tables["values"].sort_values(["public_id", "position"], kind="stable")
    labels = values.meaning.where(values.meaning != "", values.value)
    grouped = values.assign(label=labels).groupby("public_id").label.agg(list)
    listed = {
        element: value_labels(element_values)
        for element, element_values in grouped.items()
    }
    return [
        Item(
            id=row.public_id,
            canonical=without_java(row.long_name),
            type=row.datatype,
            description=without_java(cadsr_definition(row.definition)),
            lists=[("", listed[row.public_id])] if row.public_id in listed else [],
            texts=[Text(row.short_name, "short name"), *texts[row.public_id]],
            status=STATUSES.get(row.workflow_status, CURRENT),
            open=row.open,
        )
        for row in elements.itertuples()
    ]
