
BASE_TEMPLATE = """
## 1. Factual Retrieval
A query aiming to directly locate and extract clear facts, specific text snippets, or entities. The answer explicitly exists in the text.

{FactualRetrievalTemplates}

---

## 2. Complex Synthesis
A query requiring the integration, comparison, and interpretation of information across paragraphs or documents.

{ComplexSynthesisTemplates}

---

## 3. Quantitative Reasoning
A query that must be answered by performing mathematical operations or quantitative logical reasoning based on extracted base data.

{QuantitativeReasoningTemplates}

---

## 4. Multimodal Parsing
A query focusing on the multimodal and layout features of a document, requiring the extraction of specific structural or visual elements.

{MultimodalParsingTemplates}

---
"""
